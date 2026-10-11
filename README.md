# FUH-DTDA

**Fuzzy Uncertainty-aware Heterogeneous Dual Transformer Network for Drug–Disease Association Prediction**

FUH-DTDA combines drug and disease similarity graphs with a heterogeneous drug–disease–protein network to predict drug–disease associations. The model integrates graph transformers, Gaussian fuzzy confidence refinement, adaptive modality fusion, and an auxiliary uncertainty prediction head.

## Model overview

The implementation contains the following components:

- **Similarity graph encoders:** separate graph transformers learn drug and disease representations from their similarity profiles.
- **Heterogeneous graph encoder:** a heterogeneous graph transformer (HGT) processes drug, disease, and protein features and their network relationships.
- **Fuzzy confidence refinement:** learnable Gaussian membership functions produce confidence scores used in residual embedding refinement.
- **Adaptive modality fusion:** confidence-informed gates weight the similarity and heterogeneous representations before transformer-based fusion.
- **Pair interaction and prediction:** drug and disease embeddings, their element-wise product, and their absolute difference form the input to a binary classifier.
- **Auxiliary uncertainty estimation:** a sigmoid head learns an uncertainty signal using the classifier's true-class probability.

## Repository structure

```text
.
├── README.md
├── data.zip
├── data/                           # Created by extracting data.zip
│   ├── B-dataset/
│   ├── C-dataset/
│   └── F-dataset/
├── data_preprocess.py              # Data loading, sampling, folds, and graphs
├── metric.py                       # Evaluation metrics
├── train_DDA.py                    # Training and cross-validation
└── model/
    ├── FUH_DTDA_model.py           # FUH-DTDA architecture
    ├── gt_net_drug.py              # Drug similarity graph transformer
    ├── gt_net_disease.py           # Disease similarity graph transformer
    └── graph_transformer_layer.py  # Graph transformer building blocks
```

The complete `model/` directory is required in addition to the three top-level Python scripts.

## Environment and installation

The following versions are recorded in the [AMDGT README](https://github.com/JK-Liu7/AMDGT). They provide a reference environment; a complete FUH-DTDA training run has not been validated for this source snapshot.

| Dependency | Reference version |
| --- | --- |
| Python | 3.9.13 |
| CUDA toolkit | 11.3.1 |
| PyTorch | 1.10.0 |
| DGL | 0.9.0 |
| NetworkX | 2.8.4 |
| NumPy | 1.23.1 |
| scikit-learn | 0.24.2 |
| pandas | Required by the scripts; not pinned in the upstream README |

The current training script explicitly selects CUDA. Use an NVIDIA GPU supported by the chosen PyTorch/CUDA build and install a CUDA-enabled DGL package.

### 1. Create an environment

```bash
conda create -n fuh-dtda python=3.9.13 -y
conda activate fuh-dtda
```

### 2. Install dependencies

Reference commands for the CUDA 11.3 environment:

```bash
conda install pytorch==1.10.0 cudatoolkit=11.3.1 -c pytorch -c conda-forge -y
python -m pip install dgl-cu113==0.9.0 -f https://data.dgl.ai/wheels/repo.html
python -m pip install numpy==1.23.1 networkx==2.8.4 scikit-learn==0.24.2 "pandas<2"
```

The pandas range above is a suggested installation constraint, not a verified experiment version. Installation references: [PyTorch previous versions](https://docs.pytorch.org/get-started/previous-versions/) and [DGL 0.9.0](https://github.com/dmlc/dgl/releases/tag/0.9.0).

Check that both PyTorch and DGL can use CUDA:

```bash
python -c "import torch, dgl; print('PyTorch:', torch.__version__); print('DGL:', dgl.__version__); print('CUDA:', torch.cuda.is_available()); print(dgl.graph(([0], [1])).to('cuda').device)"
```

### 3. Check the model import

In the supplied source snapshot, `model/FUH_DTDA_model.py` contains:

```python
from model.v11 import gt_net_drug, gt_net_disease
```

For the repository layout shown above, replace that line with:

```python
from model import gt_net_drug, gt_net_disease
```

The `model/v11/` package is absent from this layout, so the original import prevents the training script from starting. Run all commands below from the repository root, which contains `train_DDA.py`.

## Datasets

The B-, C-, and F-datasets are public benchmark datasets distributed with [AMDGT](https://github.com/JK-Liu7/AMDGT/tree/main/data). The supplied `data.zip` contains all three datasets.

### Data preparation

Extract the archive from the repository root:

```bash
python -m zipfile -e data.zip .
```

The archive already contains a top-level `data/` directory. After extraction, the required paths are `data/B-dataset/`, `data/C-dataset/`, and `data/F-dataset/`.

### Dataset statistics

The following counts were read from the supplied archive. Association counts are data rows in `DrugDiseaseAssociationNumber.csv`, excluding its header.

| Dataset | Drugs | Diseases | Proteins | Drug–disease association rows |
| --- | ---: | ---: | ---: | ---: |
| B-dataset | 269 | 598 | 1,021 | 18,416 |
| C-dataset | 663 | 409 | 993 | 2,532 |
| F-dataset | 592 | 313 | 2,741 | 1,933 |

### Files used by the loader

| File | Content |
| --- | --- |
| `Drug_mol2vec.csv` | 300-dimensional drug embeddings |
| `DiseaseFeature.csv` | 64-dimensional disease embeddings |
| `Protein_ESM.csv` | 320-dimensional protein embeddings |
| `DrugFingerprint.csv` | Drug fingerprint similarity matrix |
| `DrugGIP.csv` | Drug Gaussian interaction profile similarity matrix |
| `DiseasePS.csv` | Disease phenotypic similarity matrix |
| `DiseaseGIP.csv` | Disease Gaussian interaction profile similarity matrix |
| `DrugDiseaseAssociationNumber.csv` | Indexed drug–disease pairs |
| `DrugProteinAssociationNumber.csv` | Indexed drug–protein pairs |
| `ProteinDiseaseAssociationNumber.csv` | Indexed disease–protein pairs, read in that order by the loader |

Keep the CSV formats and entity ordering unchanged. The loader removes the first column from similarity and feature files; embedding files are read without a header. Association files are read with a header and must contain zero-based integer node indices.

## Training

After preparing the environment, model import, and data, run:

```bash
python train_DDA.py --dataset C-dataset
```

To run the other datasets:

```bash
python train_DDA.py --dataset B-dataset
python train_DDA.py --dataset F-dataset
```

By default, training uses 10-fold stratified cross-validation, 1,000 epochs per fold, and one sampled unobserved pair per known positive association.

Example with explicit settings:

```bash
python train_DDA.py --dataset C-dataset --k_fold 10 --epochs 1000 --lr 0.0001 --weight_decay 0.001 --negative_rate 1.0 --random_seed 1234 --lambda_conf 0.1
```

To view all available arguments:

```bash
python train_DDA.py --help
```

### Main arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--dataset` | `C-dataset` | Dataset directory name under `data/` |
| `--k_fold` | `10` | Number of cross-validation folds |
| `--epochs` | `1000` | Training epochs per fold |
| `--lr` | `0.0001` | Adam learning rate |
| `--weight_decay` | `0.001` | Adam weight decay |
| `--random_seed` | `1234` | Python random seed used in sample preparation |
| `--neighbor` | `20` | Neighborhood parameter for similarity graph construction |
| `--negative_rate` | `1.0` | Requested ratio of sampled negative to positive pairs |
| `--dropout` | `0.2` | Configurable encoder, gate, and uncertainty-head dropout |
| `--lambda_conf` | `0.1` | Weight of the auxiliary uncertainty loss |

Architecture defaults are `--gt_layer 2`, `--gt_head 2`, `--gt_out_dim 200`, `--hgt_layer 2`, `--hgt_head 8`, `--hgt_in_dim 64`, `--hgt_head_dim 25`, `--tr_layer 2`, and `--tr_head 4`.

The parser also accepts `--hgt_out_dim 200`, but the model derives the HGT output width from `hgt_head * hgt_head_dim` and projects it to `gt_out_dim` when needed. The fuzzy layers use four rules by default; `--fuzzy_rules` is not exposed by the training script. The classifier MLP uses a fixed dropout of 0.4.

### Training objective

The training script combines classification and auxiliary uncertainty losses:

```text
L = CrossEntropy(logits, labels) + lambda_conf * MSE(u, 1 - p_true)
```

Here, `u` is the uncertainty-head output and `p_true` is the classifier's probability for the true class. The target `1 - p_true` is detached from gradient computation. Larger target values correspond to lower classifier probability for the true label.

## Evaluation and outputs

Each epoch reports ROC-AUC, AUPR, accuracy, precision, recall, F1-score, and Matthews correlation coefficient (MCC). AUPR is computed as the trapezoidal area under the precision–recall curve.

At the end of training, the script prints per-fold AUC and AUPR values and their means and standard deviations. For each fold, it retains the highest observed test-fold AUC and the AUPR from that same epoch.

Cross-validation splits are written to:

```text
data/<dataset>/fold/<fold_index>/data_train.csv
data/<dataset>/fold/<fold_index>/data_test.csv
```

Fold indices start at 0. Training metrics are printed to the console. The current script does not save model checkpoints or prediction files; its configured `Result/<dataset>/FUH_DTDA/` path is not used to write results.

To capture a training log:

```bash
python -u train_DDA.py --dataset C-dataset > train_C.log 2>&1
```

### Reproducibility and evaluation notes

- Unobserved drug–disease pairs are sampled as negatives; they are not experimentally confirmed negative associations.
- `--random_seed` controls Python's sample shuffling. The script does not set NumPy, PyTorch, or CUDA seeds, so this option alone does not guarantee identical training results.
- The current implementation evaluates the held-out fold every epoch and selects its best AUC. These values reflect test-fold-based epoch selection. For an independent final test estimate, select epochs using a separate validation split.
- Similarity matrices, including GIP matrices, are loaded from the supplied files and are not recomputed per fold by the current preprocessing code.

## Data availability and acknowledgements

The B-, C-, and F-datasets are available through the [AMDGT data repository](https://github.com/JK-Liu7/AMDGT/tree/main/data). FUH-DTDA source code is provided in this repository.

We acknowledge the AMDGT authors for their publicly available datasets and implementation. When using these resources, please cite the original work:

[AMDGT: Attention aware multi-modal fusion using a dual graph transformer for drug–disease associations prediction](https://www.sciencedirect.com/science/article/pii/S0950705123010778).
