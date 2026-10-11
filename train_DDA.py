import timeit
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as fn
from data_preprocess import *
from model.FUH_DTDA_model import FUH_DTDA
from metric import *

device = torch.device('cuda')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=int, default=10, help='k-fold cross validation')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight_decay')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--neighbor', type=int, default=20, help='neighbor')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')
    parser.add_argument('--dataset', default='C-dataset', help='dataset')
    parser.add_argument('--dropout', default='0.2', type=float, help='dropout')
    parser.add_argument('--gt_layer', default='2', type=int, help='graph transformer layer')
    parser.add_argument('--gt_head', default='2', type=int, help='graph transformer head')
    parser.add_argument('--gt_out_dim', default='200', type=int, help='graph transformer output dimension')
    parser.add_argument('--hgt_layer', default='2', type=int, help='heterogeneous graph transformer layer')
    parser.add_argument('--hgt_head', default='8', type=int, help='heterogeneous graph transformer head')
    parser.add_argument('--hgt_in_dim', default='64', type=int, help='heterogeneous graph transformer input dimension')
    parser.add_argument('--hgt_head_dim', default='25', type=int, help='heterogeneous graph transformer head dimension')
    parser.add_argument('--hgt_out_dim', default='200', type=int, help='heterogeneous graph transformer output dimension')
    parser.add_argument('--tr_layer', default='2', type=int, help='transformer layer')
    parser.add_argument('--tr_head', default='4', type=int, help='transformer head')
    parser.add_argument('--lambda_conf', default='0.1', type=float,
                         help='weight of the auxiliary true-class-probability confidence loss '
                              'that trains the uncertainty head')

    args = parser.parse_args()
    args.data_dir = 'data/' + args.dataset + '/'
    args.result_dir = 'Result/' + args.dataset + '/FUH_DTDA/'

    data = get_data(args)
    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']

    data = data_processing(data, args)
    data = k_fold(data, args)

    drdr_graph, didi_graph, data = dgl_similarity_graph(data, args)

    drdr_graph = drdr_graph.to(device)
    didi_graph = didi_graph.to(device)

    drug_feature = torch.FloatTensor(data['drugfeature']).to(device)
    disease_feature = torch.FloatTensor(data['diseasefeature']).to(device)
    protein_feature = torch.FloatTensor(data['proteinfeature']).to(device)
    all_sample = torch.tensor(data['all_drdi']).long()

    start = timeit.default_timer()

    cross_entropy = nn.CrossEntropyLoss()

    Metric = ('Epoch\t\tTime\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')
    AUCs, AUPRs = [], []

    print('Dataset:', args.dataset)

    for i in range(args.k_fold):

        print('fold:', i)
        print(Metric)

        model = FUH_DTDA(args, return_aux=True)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

        best_auc, best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = 0, 0, 0, 0, 0, 0, 0
        X_train = torch.LongTensor(data['X_train'][i]).to(device)
        Y_train = torch.LongTensor(data['Y_train'][i]).to(device)
        X_test = torch.LongTensor(data['X_test'][i]).to(device)
        Y_test = data['Y_test'][i].flatten()

        drdipr_graph, data = dgl_heterograph(data, data['X_train'][i], args)
        drdipr_graph = drdipr_graph.to(device)

        for epoch in range(args.epochs):
            model.train()
            _, train_score, train_aux = model(drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, X_train)

            train_label = torch.flatten(Y_train)
            ce_loss = cross_entropy(train_score, train_label)

            # Auxiliary loss for the uncertainty head. The target is
            # 1 - true-class-probability (TCP) of the classifier, so the head
            # is trained to output HIGH values when the model is likely wrong
            # and LOW values when it is likely right - matching the literal
            # meaning of "uncertainty" (as opposed to regressing TCP itself,
            # which would make it a confidence score, not an uncertainty one).
            # The TCP-regression target is inspired by Corbière et al.,
            # NeurIPS 2019 ("Addressing Failure Prediction by Learning Model
            # Confidence"), but trained jointly with the classifier every
            # step (not their two-stage decoupled scheme: train classifier
            # to convergence, freeze it, then train the confidence head
            # alone). The uncertainty head reads pair_embedding.detach() in
            # FUH_DTDA_model.forward(), so this loss only updates the head's
            # own weights and cannot distort the shared representation.
            # Without this term the uncertainty head receives no gradient at
            # all, since its output is not otherwise used anywhere in the loss.
            train_probs = fn.softmax(train_score.detach(), dim=-1)
            tcp_target = train_probs.gather(1, train_label.unsqueeze(1)).squeeze(1)
            uncertainty_target = 1.0 - tcp_target
            conf_loss = fn.mse_loss(train_aux['uncertainty'].squeeze(-1), uncertainty_target)

            train_loss = ce_loss + args.lambda_conf * conf_loss
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                dr_representation, test_score, test_aux = model(drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, X_test)

            test_prob = fn.softmax(test_score, dim=-1)
            test_score = torch.argmax(test_score, dim=-1)

            test_prob = test_prob[:, 1]
            test_prob = test_prob.cpu().numpy()

            test_score = test_score.cpu().numpy()

            AUC, AUPR, accuracy, precision, recall, f1, mcc = get_metric(Y_test, test_score, test_prob)

            end = timeit.default_timer()
            time = end - start
            show = [epoch + 1, round(time, 2), round(AUC, 5), round(AUPR, 5), round(accuracy, 5),
                       round(precision, 5), round(recall, 5), round(f1, 5), round(mcc, 5)]
            print('\t\t'.join(map(str, show)))
            if AUC > best_auc:
                best_epoch = epoch + 1
                best_auc = AUC
                best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = AUPR, accuracy, precision, recall, f1, mcc
                print('AUC improved at epoch ', best_epoch, ';\tbest_auc:', best_auc)

        AUCs.append(best_auc)
        AUPRs.append(best_aupr)

    print('AUC:', AUCs)
    AUC_mean = np.mean(AUCs)
    AUC_std = np.std(AUCs)
    print('Mean AUC:', AUC_mean, '(', AUC_std, ')')

    print('AUPR:', AUPRs)
    AUPR_mean = np.mean(AUPRs)
    AUPR_std = np.std(AUPRs)
    print('Mean AUPR:', AUPR_mean, '(', AUPR_std, ')')



