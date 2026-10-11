# FUH_DTDA_model.py
# Fuzzy Uncertainty-aware Heterogeneous Dual Transformer Network
# for Drug–Disease Association Prediction
#
# This file is designed as a drop-in alternative to AMNTDDA.py.
# It keeps the same main inputs in forward():
#   drdr_graph, didi_graph, drdipr_graph,
#   drug_feature, disease_feature, protein_feature, sample
#
# Main changes compared with AMNTDDA:
#   1. Safe CUDA/CPU device handling
#   2. Fuzzy confidence refinement for Graph Transformer embeddings
#   3. Fuzzy confidence refinement for HGT embeddings
#   4. Fuzzy adaptive modality gate before Transformer fusion
#   5. Rich drug–disease pair representation
#   6. Optional uncertainty head for prediction confidence

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch

from model.v11 import gt_net_drug, gt_net_disease


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FuzzyConfidenceLayer(nn.Module):
    """
    Gaussian fuzzy membership layer for embedding confidence estimation.

    Input:
        x: [N, D]

    Output:
        x_refined: [N, D]
        confidence: [N]
        membership: [N, R]

    Meaning:
        Each embedding vector is compared with learnable fuzzy centers.
        The resulting fuzzy memberships are converted into a confidence score.
        Low-confidence embeddings are softly down-weighted.
    """

    def __init__(self, dim, num_rules=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_rules = num_rules

        self.centers = nn.Parameter(torch.randn(num_rules, dim) * 0.1)
        self.log_sigma = nn.Parameter(torch.zeros(num_rules, dim))

        self.rule_to_confidence = nn.Sequential(
            nn.Linear(num_rules, num_rules),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_rules, 1)
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [N, D]
        sigma = F.softplus(self.log_sigma) + 1e-6

        x_expand = x.unsqueeze(1)              # [N, 1, D]
        centers = self.centers.unsqueeze(0)    # [1, R, D]
        sigma = sigma.unsqueeze(0)             # [1, R, D]

        # Gaussian fuzzy membership: [N, R]
        membership = torch.exp(
            -torch.mean((x_expand - centers) ** 2 / (2.0 * sigma ** 2), dim=-1)
        )

        # Confidence score in [0, 1]: [N]
        confidence = torch.sigmoid(self.rule_to_confidence(membership)).squeeze(-1)

        # Residual fuzzy refinement
        x_refined = x * confidence.unsqueeze(-1)
        x_refined = self.norm(x + x_refined)

        return x_refined, confidence, membership


class FuzzyModalityGate(nn.Module):
    """
    Fuzzy adaptive gate for two modalities:
        1. similarity-based embedding
        2. heterogeneous-network embedding

    Input:
        h_sim: [N, D]
        h_hgt: [N, D]

    Output:
        stacked embeddings: [N, 2, D]
        gate weights: [N, 2]

    Meaning:
        The model learns how much it should trust each modality for each node.
    """

    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 2)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, h_sim, h_hgt, conf_sim=None, conf_hgt=None):
        # Interaction-aware gate input
        z = torch.cat(
            [
                h_sim,
                h_hgt,
                h_sim * h_hgt,
                torch.abs(h_sim - h_hgt)
            ],
            dim=-1
        )

        logits = self.gate(z)  # [N, 2]

        # Inject fuzzy confidence into modality logits if available
        if conf_sim is not None and conf_hgt is not None:
            conf = torch.stack([conf_sim, conf_hgt], dim=-1)  # [N, 2]
            logits = logits + torch.log(conf + 1e-6)

        weights = torch.softmax(logits, dim=-1)  # [N, 2]

        h_sim_w = h_sim * weights[:, 0].unsqueeze(-1)
        h_hgt_w = h_hgt * weights[:, 1].unsqueeze(-1)

        h_sim_w = self.norm(h_sim + h_sim_w)
        h_hgt_w = self.norm(h_hgt + h_hgt_w)

        return torch.stack([h_sim_w, h_hgt_w], dim=1), weights


class PairInteractionLayer(nn.Module):
    """
    Build a rich drug–disease pair representation.

    Instead of using only element-wise product, this layer uses:
        [drug || disease || drug * disease || |drug - disease|]

    Input:
        drug_emb: [B, D]
        disease_emb: [B, D]

    Output:
        pair_emb: [B, 4D]
    """

    def forward(self, drug_emb, disease_emb):
        return torch.cat(
            [
                drug_emb,
                disease_emb,
                drug_emb * disease_emb,
                torch.abs(drug_emb - disease_emb)
            ],
            dim=-1
        )


class FUH_DTDA(nn.Module):
    """
    FUH-DTDA:
    Fuzzy Uncertainty-aware Heterogeneous Dual Transformer Network
    for Drug–Disease Association Prediction.

    This class can replace AMNTDDA in the training script, with minor adjustment:
        model = FUH_DTDA(args)

    Forward output:
        return drug_embedding, logits

    If return_aux=True:
        return drug_embedding, logits, aux_info
    """

    def __init__(self, args, return_aux=False):
        super().__init__()
        self.args = args
        self.return_aux = return_aux
        self.device = get_device()

        # =========================
        # 1. Feature projection
        # =========================
        self.drug_linear = nn.Linear(300, args.hgt_in_dim)
        self.protein_linear = nn.Linear(320, args.hgt_in_dim)

        # Optional projection for disease feature.
        # If disease_feature is already hgt_in_dim, this layer will be used only when needed in forward().
        self.disease_linear = None

        # =========================
        # 2. Dual Graph Transformer
        # =========================
        self.gt_drug = gt_net_drug.GraphTransformer(
            self.device,
            args.gt_layer,
            args.drug_number,
            args.gt_out_dim,
            args.gt_out_dim,
            args.gt_head,
            args.dropout
        )

        self.gt_disease = gt_net_disease.GraphTransformer(
            self.device,
            args.gt_layer,
            args.disease_number,
            args.gt_out_dim,
            args.gt_out_dim,
            args.gt_head,
            args.dropout
        )

        # =========================
        # 3. Heterogeneous Graph Transformer
        # =========================
        self.hgt = nn.ModuleList()

        for _ in range(args.hgt_layer - 1):
            self.hgt.append(
                dgl.nn.pytorch.conv.HGTConv(
                    args.hgt_in_dim,
                    int(args.hgt_in_dim / args.hgt_head),
                    args.hgt_head,
                    3,  # number of node types: drug, disease, protein
                    3,  # number of edge types
                    args.dropout
                )
            )

        self.hgt.append(
            dgl.nn.pytorch.conv.HGTConv(
                args.hgt_in_dim,
                args.hgt_head_dim,
                args.hgt_head,
                3,
                3,
                args.dropout
            )
        )

        # hgt_out_dim should match gt_out_dim.
        # Example: hgt_head=8, hgt_head_dim=25 => 200.
        self.hgt_out_dim = args.hgt_head * args.hgt_head_dim

        if self.hgt_out_dim != args.gt_out_dim:
            self.hgt_to_gt = nn.Linear(self.hgt_out_dim, args.gt_out_dim)
        else:
            self.hgt_to_gt = nn.Identity()

        # =========================
        # 4. Fuzzy confidence modules
        # =========================
        num_rules = getattr(args, "fuzzy_rules", 4)

        self.fuzzy_drug_sim = FuzzyConfidenceLayer(
            args.gt_out_dim, num_rules=num_rules, dropout=args.dropout
        )
        self.fuzzy_disease_sim = FuzzyConfidenceLayer(
            args.gt_out_dim, num_rules=num_rules, dropout=args.dropout
        )
        self.fuzzy_drug_hgt = FuzzyConfidenceLayer(
            args.gt_out_dim, num_rules=num_rules, dropout=args.dropout
        )
        self.fuzzy_disease_hgt = FuzzyConfidenceLayer(
            args.gt_out_dim, num_rules=num_rules, dropout=args.dropout
        )

        # =========================
        # 5. Fuzzy adaptive modality fusion
        # =========================
        self.drug_gate = FuzzyModalityGate(args.gt_out_dim, dropout=args.dropout)
        self.disease_gate = FuzzyModalityGate(args.gt_out_dim, dropout=args.dropout)

        encoder_layer_drug = nn.TransformerEncoderLayer(
            d_model=args.gt_out_dim,
            nhead=args.tr_head,
            dropout=args.dropout,
            batch_first=True
        )
        encoder_layer_disease = nn.TransformerEncoderLayer(
            d_model=args.gt_out_dim,
            nhead=args.tr_head,
            dropout=args.dropout,
            batch_first=True
        )

        self.drug_trans = nn.TransformerEncoder(
            encoder_layer_drug,
            num_layers=args.tr_layer
        )
        self.disease_trans = nn.TransformerEncoder(
            encoder_layer_disease,
            num_layers=args.tr_layer
        )

        self.final_drug_norm = nn.LayerNorm(args.gt_out_dim * 2)
        self.final_disease_norm = nn.LayerNorm(args.gt_out_dim * 2)

        # =========================
        # 6. Pair interaction + prediction
        # =========================
        self.pair_interaction = PairInteractionLayer()

        final_node_dim = args.gt_out_dim * 2
        pair_dim = final_node_dim * 4

        self.mlp = nn.Sequential(
            nn.Linear(pair_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

        # Optional uncertainty head.
        # It returns a confidence/uncertainty signal for analysis or auxiliary loss.
        self.uncertainty_head = nn.Sequential(
            nn.Linear(pair_dim, 256),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def _maybe_project_disease(self, disease_feature):
        """
        Disease feature in AMNTDDA is assumed to already be hgt_in_dim.
        This function makes the model safer if disease_feature has another dimension.
        """
        if disease_feature.shape[-1] == self.args.hgt_in_dim:
            return disease_feature

        if self.disease_linear is None:
            self.disease_linear = nn.Linear(
                disease_feature.shape[-1],
                self.args.hgt_in_dim
            ).to(disease_feature.device)

        return self.disease_linear(disease_feature)

    def forward(
        self,
        drdr_graph,
        didi_graph,
        drdipr_graph,
        drug_feature,
        disease_feature,
        protein_feature,
        sample
    ):
        # =========================
        # 1. Similarity graph embeddings
        # =========================
        dr_sim = self.gt_drug(drdr_graph)       # [num_drug, gt_out_dim]
        di_sim = self.gt_disease(didi_graph)   # [num_disease, gt_out_dim]

        # Fuzzy confidence refinement on similarity embeddings
        dr_sim, dr_sim_conf, dr_sim_membership = self.fuzzy_drug_sim(dr_sim)
        di_sim, di_sim_conf, di_sim_membership = self.fuzzy_disease_sim(di_sim)

        # =========================
        # 2. Heterogeneous graph embeddings
        # =========================
        drug_feature = self.drug_linear(drug_feature)
        disease_feature = self._maybe_project_disease(disease_feature)
        protein_feature = self.protein_linear(protein_feature)

        feature_dict = {
            "drug": drug_feature,
            "disease": disease_feature,
            "protein": protein_feature
        }

        drdipr_graph.ndata["h"] = feature_dict
        g = dgl.to_homogeneous(drdipr_graph, ndata="h")

        feature = torch.cat(
            [drug_feature, disease_feature, protein_feature],
            dim=0
        )

        hgt_out = feature
        for layer in self.hgt:
            hgt_out = layer(
                g,
                hgt_out,
                g.ndata["_TYPE"],
                g.edata["_TYPE"],
                presorted=True
            )

        dr_hgt = hgt_out[:self.args.drug_number, :]
        di_hgt = hgt_out[
            self.args.drug_number:self.args.drug_number + self.args.disease_number,
            :
        ]

        dr_hgt = self.hgt_to_gt(dr_hgt)
        di_hgt = self.hgt_to_gt(di_hgt)

        # Fuzzy confidence refinement on HGT embeddings
        dr_hgt, dr_hgt_conf, dr_hgt_membership = self.fuzzy_drug_hgt(dr_hgt)
        di_hgt, di_hgt_conf, di_hgt_membership = self.fuzzy_disease_hgt(di_hgt)

        # =========================
        # 3. Fuzzy adaptive modality fusion
        # =========================
        dr_seq, dr_gate_weight = self.drug_gate(
            dr_sim,
            dr_hgt,
            dr_sim_conf,
            dr_hgt_conf
        )
        di_seq, di_gate_weight = self.disease_gate(
            di_sim,
            di_hgt,
            di_sim_conf,
            di_hgt_conf
        )

        # Transformer fusion: [N, 2, D]
        dr_fused = self.drug_trans(dr_seq)
        di_fused = self.disease_trans(di_seq)

        # Flatten two modality tokens into final embedding
        dr = dr_fused.reshape(self.args.drug_number, 2 * self.args.gt_out_dim)
        di = di_fused.reshape(self.args.disease_number, 2 * self.args.gt_out_dim)

        dr = self.final_drug_norm(dr)
        di = self.final_disease_norm(di)

        # =========================
        # 4. Drug–disease pair prediction
        # =========================
        drug_pair = dr[sample[:, 0]]
        disease_pair = di[sample[:, 1]]

        pair_embedding = self.pair_interaction(drug_pair, disease_pair)

        logits = self.mlp(pair_embedding)
        uncertainty = self.uncertainty_head(pair_embedding)

        if self.return_aux:
            aux_info = {
                "drug_embedding": dr,
                "disease_embedding": di,
                "drug_sim_conf": dr_sim_conf,
                "disease_sim_conf": di_sim_conf,
                "drug_hgt_conf": dr_hgt_conf,
                "disease_hgt_conf": di_hgt_conf,
                "drug_gate_weight": dr_gate_weight,
                "disease_gate_weight": di_gate_weight,
                "uncertainty": uncertainty,
                "drug_sim_membership": dr_sim_membership,
                "disease_sim_membership": di_sim_membership,
                "drug_hgt_membership": dr_hgt_membership,
                "disease_hgt_membership": di_hgt_membership
            }
            return dr, logits, aux_info

        # Keep the same output style as AMNTDDA:
        # return drug embedding and classification logits.
        return dr, logits


# Optional alias, so old code can use:
#   from FUH_DTDA_model import AMNTDDA
# without changing the training script too much.
AMNTDDA = FUH_DTDA
