import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
# from layers import GraphAttentionLayer, SpGraphAttentionLayer,GraphConvolutionLayer,SpResidualAttentionLayer

class AttentionFusion(nn.Module):
    def __init__(self, n_h):
        super(AttentionFusion, self).__init__()
        self.att = nn.Sequential(
            nn.Linear(n_h * 2, n_h),
            nn.Tanh(),
            nn.Linear(n_h, 2), # 输出两个权值 α1, α2
            nn.Softmax(dim=-1)
        )

    def forward(self, h_v, z_m):
        # fv = α1hv + α2z(m)v
        cat_feat = torch.cat([h_v, z_m], dim=-1)
        alpha = self.att(cat_feat) # (batch, nodes, 2)
        f_v = alpha[:, :, 0:1] * h_v + alpha[:, :, 1:2] * z_m
        return f_v, alpha

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
            
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)

class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq,1).values

class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values

class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0,2,1)
        sim = torch.matmul(seq,query)
        sim = F.softmax(sim,dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq,sim)
        out = torch.sum(out,1)
        return out


class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        # negative
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1,:], c_mi[:-1,:]),0)
            scs.append(self.f_k(h_pl, c_mi))

        logits = torch.cat(tuple(scs))

        return logits
class Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout):
        super(Structure_Decoder, self).__init__()

        self.gc1 = GCN(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T

        return x

class Curvature_Decoder(nn.Module):
    def __init__(self, n_h):
        super(Curvature_Decoder, self).__init__()
        # 公式 (18): 两层 MLP 作用于拼接的嵌入 [fu || fv]
        self.mlp = nn.Sequential(
            nn.Linear(n_h * 2, n_h),
            nn.ReLU(),
            nn.Linear(n_h, 1)
        )

    def forward(self, fu, fv):
        # 拼接节点对的嵌入
        combined = torch.cat((fu, fv), dim=-1)
        return self.mlp(combined).squeeze(-1)

# class Model(nn.Module):
#     def __init__(self, n_in, n_h, activation, negsamp_round, readout,motif_size,hidden1,hidden2,alpha,dropout=0.2):
#         super(Model, self).__init__()
#         self.read_mode = readout
#         self.alpha = alpha
#         self.dropout = dropout
#         self.gcn = GCN(n_in, n_h, activation)
#         self.dec = GCN(n_h, n_in, activation)
#         self.gc_enc1 = GCN(motif_size, n_h,activation)
#         self.gc_dec1 = GCN(n_h, motif_size,activation)
#         self.pdist = nn.PairwiseDistance(p=2)
#
#         if readout == 'max':
#             self.read = MaxReadout()
#         elif readout == 'min':
#             self.read = MinReadout()
#         elif readout == 'avg':
#             self.read = AvgReadout()
#         elif readout == 'weighted_sum':
#             self.read = WSReadout()
#
#         self.disc = Discriminator(n_h, negsamp_round)
#
#     def forward(self, seq1,seq2, adj,adjm,motifs, sparse=False):
#
#
#         s_0 = self.gc_enc1(motifs,adjm,sparse)
#         s_1 = self.gc_dec1(s_0,adjm,sparse)
#         h_1 = self.gcn(seq1, adj, sparse)
#         f_1 = self.gcn(seq2, adj, sparse)
#
#         f_2 = self.dec(f_1,adj,sparse)
#         if self.read_mode != 'weighted_sum':
#             c = self.read(h_1[:,: -1,:])
#             h_mv = h_1[:,-1,:]
#         else:
#             h_mv = h_1[:, -1, :]
#             c = self.read(h_1[:,: -1,:], h_1[:,-2: -1, :])
#
#         ret = self.disc(c, h_mv)
#
#         return ret,s_1,f_2
#
#     def inference(self, seq1,seq2,adj, adjm, motifs, alpha,sparse=False):
#         s_0 = self.gc_enc1(motifs, adjm, sparse)
#         s_1 = self.gc_dec1(s_0, adjm, sparse)
#         h_1 = self.gcn(seq1, adj, sparse)
#
#         f_0 = self.gcn(seq2,adj,sparse)
#         f_1 = self.dec(f_0,adj,sparse)
#         dist0 = self.pdist(s_1[:, -2, :], motifs[:, -1, :])
#         dist1 = self.pdist(f_1[:, -2, :], seq2[:, -1, :])
#         dist = alpha*dist0+(1-alpha)*dist1
#
#         if self.read_mode != 'weighted_sum':
#             c = self.read(h_1[:,: -1,:])
#             h_mv = h_1[:,-1,:]
#         else:
#             h_mv = h_1[:, -1, :]
#             c = self.read(h_1[:,: -1,:], h_1[:,-2: -1, :])
#
#         ret = self.disc(c, h_mv)
#
#         return ret,dist,dist0,dist1


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout, motif_size, hidden1, hidden2, alpha, dropout=0.2):
        super(Model, self).__init__()
        self.read_mode = readout # 确保保存了 readout 模式
        self.gcn_sem = GCN(n_in, n_h, activation)
        self.gc_enc_str = GCN(motif_size, n_h, activation)

        self.fusion = AttentionFusion(n_h)
        self.curv_dec = Curvature_Decoder(n_h)

        self.dec_sem = GCN(n_h, n_in, activation)
        self.gc_dec_str = GCN(n_h, motif_size, activation)

        self.disc = Discriminator(n_h, negsamp_round)
        self.pdist = nn.PairwiseDistance(p=2) # 用于计算重构距离

        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

    def forward(self, seq_sem, seq_raw, adj, adjm, motifs, sparse=False):
        # 1. 提取语义与结构特征
        h_v = self.gcn_sem(seq_sem, adj, sparse)
        z_m = self.gc_enc_str(motifs, adjm, sparse)

        # 2. 注意力融合
        f_v, att_weights = self.fusion(h_v, z_m)

        # 3. 对比学习任务 (基于融合特征)
        c = self.read(f_v[:, :-1, :])
        ret = self.disc(c, f_v[:, -1, :])

        # 4. 重构任务
        s_1 = self.gc_dec_str(f_v, adjm, sparse) # 结构重构
        f_2 = self.dec_sem(f_v, adj, sparse)    # 语义重构

        return ret, s_1, f_2, f_v

    # 新增的 inference 方法，适配 run.py 的调用
    def inference(self, seq_sem, seq_raw, adj, adjm, motifs, alpha, sparse=False):
        # 1. 执行前向传播获取所有中间变量
        logits, s_1, f_2, f_v = self.forward(seq_sem, seq_raw, adj, adjm, motifs, sparse)

        # 2. 计算结构重构误差 (rmotif)
        # 对应 run.py 中 s_1[:, -2, :] 与 motifs[:, -1, :] 的距离
        dist_s = torch.mean(torch.pow(s_1[:, -2, :] - motifs[:, -1, :], 2), dim=1)

        # 3. 计算语义重构误差 (rsem)
        # 对应 run.py 中 f_2[:, -2, :] 与 seq_raw[:, -1, :] 的距离
        dist_f = torch.mean(torch.pow(f_2[:, -2, :] - seq_raw[:, -1, :], 2), dim=1)

        # 4. 根据 alpha 融合重构得分
        dist_combined = alpha * dist_s + (1 - alpha) * dist_f

        # 返回 logits(对比学习得分), dist_combined(重构得分), f_v(用于曲率计算)
        return logits, dist_combined, f_v