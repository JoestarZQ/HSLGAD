
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from model import Model
from utils import *
import copy
from sklearn.metrics import roc_auc_score
import random
import os
import dgl

import argparse
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set argument
parser = argparse.ArgumentParser(description='HSLGAD:Hybrid Self-Supervised Learning Based on Higher-Order Structures for Graph Anomaly Detection')
parser.add_argument('--dataset', type=str, default='citeseer')  # 'BlogCatalog'    'ACM'  'cora'  'citeseer'  'pubmed'
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--embedding_dim', type=int, default=32)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg') 
parser.add_argument('--auc_test_rounds', type=int, default=1)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--alpha', type=float, default=0.7, help='balance parameter')
parser.add_argument('--beta', type=float, default=0.6, help='loss parameter')
parser.add_argument('--gamma', type=float, default=0.3, help='weight for curvature loss') # 新增几何权重
parser.add_argument('--cuda', type= bool,default=True, help='Use CUDA if available')
args = parser.parse_args()

if args.lr is None:
    if args.dataset in ['cora','citeseer','pubmed','Flickr']:
        args.lr = 1e-3
    elif args.dataset == 'ACM':
        args.lr = 5e-4
    elif args.dataset == 'BlogCatalog':
        args.lr = 3e-3

if args.num_epoch is None:
    if args.dataset in ['cora','citeseer','pubmed']:
        args.num_epoch = 150
    elif args.dataset in ['BlogCatalog','Flickr','ACM']:
        args.num_epoch = 400

batch_size = args.batch_size
subgraph_size = args.subgraph_size

print('Dataset: ',args.dataset)


# Set random seed
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if args.cuda else "cpu")
print(device)

adj, adj_norm,features, labels, idx_train, idx_val,\
idx_test, ano_label, str_ano_label, attr_ano_label,motifs,= load_mat(args.dataset)

print("Computing Ricci Curvature...")
curvature_matrix = compute_ricci_curvature(adj)

raw_features = features.todense()
features, _ = preprocess_features(features)
dgl_graph = adj_to_dgl_graph(adj)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]
motifs_size = motifs.shape[1]

adj = normalize_adj(adj)
adj = (adj + sp.eye(adj.shape[0])).todense()#A = A + I

features = torch.FloatTensor(features[np.newaxis])
raw_features = torch.FloatTensor(raw_features[np.newaxis])
adj = torch.FloatTensor(adj[np.newaxis])

labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

scaler2 = MinMaxScaler()
motifs_norm = scaler2.fit_transform(motifs)
motifs_norm = torch.FloatTensor(motifs_norm[np.newaxis]).to(device)
adj_norm = torch.FloatTensor(adj_norm).to(device)



emb_size=args.embedding_dim
hidden1=emb_size*2
hidden2=hidden1*2



model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout,motifs_size, hidden1,hidden2,args.dropout,args.alpha).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if torch.cuda.is_available():
    print('Using CUDA')
    model.to(device)
    features = features.to(device)
    raw_features = raw_features.to(device)  # 新增这一行
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

if torch.cuda.is_available():
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(device))
else:
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()

cnt_wait = 0
best = 1e9
best_t = 0
batch_num = nb_nodes // batch_size + 1


added_adj_zero_row = torch.zeros((nb_nodes, 1, subgraph_size))

added_adj_zero_col = torch.zeros((nb_nodes, subgraph_size + 1, 1))
added_adj_zero_col[:,-1,:] = 1.
added_feat_zero_row = torch.zeros((nb_nodes, 1, ft_size))
if torch.cuda.is_available():
    added_adj_zero_row = added_adj_zero_row.to(device)
    added_adj_zero_col = added_adj_zero_col.to(device)
    added_feat_zero_row = added_feat_zero_row.to(device)


mse_loss = nn.MSELoss(reduction='mean')

with tqdm(total=args.num_epoch) as pbar:
    pbar.set_description('Training')
    for epoch in range(args.num_epoch):
        loss_full_batch = torch.zeros((nb_nodes,1))
        if torch.cuda.is_available():
            loss_full_batch = loss_full_batch.to(device)

        model.train()

        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        total_loss = 0.

        subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)



        for batch_idx in range(batch_num):

            optimiser.zero_grad()

            is_final_batch = (batch_idx == (batch_num - 1))

            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]


            cur_batch_size = len(idx)

            lbl = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))), 1)
            ba = []
            bf = []
            bm = []
            bam = []
            raw_bf =[]
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
            added_adj_zero_col[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

            if torch.cuda.is_available():
                lbl = lbl.to(device)
                added_adj_zero_row = added_adj_zero_row.to(device)
                added_adj_zero_col = added_adj_zero_col.to(device)
                added_feat_zero_row = added_feat_zero_row.to(device)

            for i in idx:
                cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                cur_feat = features[:, subgraphs[i], :]
                cur_motif = motifs_norm[:, subgraphs[i], :]
                cur_adjm = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                raw_cur_feat = raw_features[:, subgraphs[i], :]
                ba.append(cur_adj)
                bf.append(cur_feat)
                bm.append(cur_motif)
                bam.append(cur_adjm)
                raw_bf.append(raw_cur_feat)

                # ========= 替换的部分开始 =========
                # 新增：定义 motif 维度的全零占位符，用于维度对齐
            added_motif_zero_row = torch.zeros((cur_batch_size, 1, motifs_size))
            if torch.cuda.is_available():
                added_motif_zero_row = added_motif_zero_row.to(device)

            ba = torch.cat(ba)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)

            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

            raw_bf = torch.cat(raw_bf)
            raw_bf = torch.cat((raw_bf[:, :-1, :], added_feat_zero_row, raw_bf[:, -1:, :]), dim=1)

            # 修复 1：对 bam(结构邻接矩阵) 进行与 ba 相同的扩展
            bam = torch.cat(bam)
            bam = torch.cat((bam, added_adj_zero_row), dim=1)
            bam = torch.cat((bam, added_adj_zero_col), dim=2)

            # 修复 2：对 bm(结构特征矩阵) 进行与 bf 相同的扩展
            bm = torch.cat(bm)
            bm = torch.cat((bm[:, :-1, :], added_motif_zero_row, bm[:, -1:, :]), dim=1)

            # 现在维度已经对齐，可以传入模型
            logits, s_1, f_2, f_v = model(bf, raw_bf, ba, bam, bm)
            loss_all = b_xent(logits, lbl)

            loss_cl = torch.mean(loss_all)
            loss_re = args.alpha * mse_loss(s_1[:, -2, :], bm[:, -1, :]) + \
                      (1 - args.alpha) * mse_loss(f_2[:, -2, :], raw_bf[:, -1, :])

            kappa_hat = model.curv_dec(f_v[:, -1, :], f_v[:, -2, :])
            target_kappa = get_batch_curvature(idx, subgraphs, curvature_matrix).to(device)
            loss_curv = mse_loss(kappa_hat, target_kappa)

            loss = (1 - args.beta - args.gamma) * loss_cl + args.beta * loss_re + args.gamma * loss_curv
            # ========= 替换的部分结束 =========

            loss.backward()
            optimiser.step()

            loss = loss.detach().cpu().numpy()
            loss_full_batch[idx] = loss_all[: cur_batch_size].detach()

            if not is_final_batch:
                total_loss += loss

        mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes

        if mean_loss < best:
            best = mean_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_model.pkl')
        else:
            cnt_wait += 1

        pbar.set_postfix(loss=mean_loss)
        pbar.update(1)



print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('best_model.pkl'))

multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
multi_round_ano_score_p = np.zeros((args.auc_test_rounds, nb_nodes))
multi_round_ano_score_n = np.zeros((args.auc_test_rounds, nb_nodes))


with tqdm(total=args.auc_test_rounds) as pbar_test:
    pbar_test.set_description('Testing')
    for round in range(args.auc_test_rounds):

        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)

        subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)


        for batch_idx in range(batch_num):

            optimiser.zero_grad()

            is_final_batch = (batch_idx == (batch_num - 1))

            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]

            cur_batch_size = len(idx)

            ba = []
            bf = []
            bm = []
            bam = []
            raw_bf = []
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
            added_adj_zero_col[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

            if torch.cuda.is_available():
                lbl = lbl.to(device)
                added_adj_zero_row = added_adj_zero_row.to(device)
                added_adj_zero_col = added_adj_zero_col.to(device)
                added_feat_zero_row = added_feat_zero_row.to(device)

            for i in idx:
                cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                cur_feat = features[:, subgraphs[i], :]
                cur_motif = motifs_norm[:, subgraphs[i], :]
                cur_adjm = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                raw_cur_feat = raw_features[:, subgraphs[i], :]
                bm.append(cur_motif)
                ba.append(cur_adj)
                bf.append(cur_feat)
                bam.append(cur_adjm)
                raw_bf.append(raw_cur_feat)

                # ========= 替换的部分开始 =========
            added_motif_zero_row = torch.zeros((cur_batch_size, 1, motifs_size))
            if torch.cuda.is_available():
                added_motif_zero_row = added_motif_zero_row.to(device)

            ba = torch.cat(ba)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)

            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

            raw_bf = torch.cat(raw_bf)
            raw_bf = torch.cat((raw_bf[:, :-1, :], added_feat_zero_row, raw_bf[:, -1:, :]), dim=1)

            # 修复 1：bam 扩展
            bam = torch.cat(bam)
            bam = torch.cat((bam, added_adj_zero_row), dim=1)
            bam = torch.cat((bam, added_adj_zero_col), dim=2)

            # 修复 2：bm 扩展
            bm = torch.cat(bm)
            bm = torch.cat((bm[:, :-1, :], added_motif_zero_row, bm[:, -1:, :]), dim=1)

            with torch.no_grad():
                logits, dist_combined, f_v = model.inference(bf, raw_bf, ba, bam, bm, args.alpha)
                logits = torch.sigmoid(logits)

            k_pred = model.curv_dec(torch.cat([f_v[:, -1, :], f_v[:, -2, :]], dim=-1)).squeeze()
            k_true = get_batch_curvature(idx, subgraphs, curvature_matrix).to(device)
            r_curv = torch.pow(k_true - k_pred, 2).cpu().numpy()

            s_cl = - (logits[:cur_batch_size] - logits[cur_batch_size:]).cpu().numpy()
            s_re = dist_combined.cpu().numpy()
            s_geo = r_curv

            scaler1 = MinMaxScaler()
            scaler2 = MinMaxScaler()
            scaler3 = MinMaxScaler()

            s_1 = scaler1.fit_transform(s_cl.reshape(-1, 1)).flatten()
            s_2 = scaler2.fit_transform(s_re.reshape(-1, 1)).flatten()
            s_3 = scaler3.fit_transform(s_geo.reshape(-1, 1)).flatten()

            ano_score = (1 - args.beta - args.gamma) * s_1 + args.beta * s_2 + args.gamma * s_3
            # ========= 替换的部分结束 =========

            multi_round_ano_score[round, idx] = ano_score

        pbar_test.update(1)

ano_score_final = np.mean(multi_round_ano_score, axis=0)

auc = roc_auc_score(ano_label, ano_score_final)

print('AUC:{:.4f}'.format(auc))



