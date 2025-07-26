import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool
from torch_geometric.utils import add_self_loops
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
import scipy.signal as sig

from config import ADAPTATION, ADAPT_N, TRAINING_DATA, GAT_MODEL_PT

# Load training data once
_tr = pickle.load(open(TRAINING_DATA, 'rb'))
n_channels = _tr['data'].shape[2]

def plvfcn(eegData: np.ndarray) -> np.ndarray:
    phase = np.angle(sig.hilbert(eegData, axis=0))
    C = eegData.shape[1]
    plv = np.zeros((C, C))
    for i in range(C):
        for j in range(i+1, C):
            d = phase[:, j] - phase[:, i]
            plv[i, j] = plv[j, i] = abs(np.exp(1j*d).mean())
    return plv

def threshold_graph_edges(plv, topk_percent=0.4):
    """Returns thresholded edge_index for top X% of PLV connections."""
    plv = plv.copy()
    np.fill_diagonal(plv, 0.0)
    triu_indices = np.triu_indices(plv.shape[0], k=1)
    edge_weights = plv[triu_indices]

    k = int(len(edge_weights) * topk_percent)
    if k == 0:
        k = 1  # Ensure at least one edge

    topk_indices = np.argpartition(edge_weights, -k)[-k:]
    row = triu_indices[0][topk_indices]
    col = triu_indices[1][topk_indices]

    edge_index = np.hstack([
        np.stack([row, col], axis=0),
        np.stack([col, row], axis=0)
    ])
    edge_index, _ = add_self_loops(torch.tensor(edge_index, dtype=torch.long), num_nodes=plv.shape[0])
    return edge_index

class SimpleGAT(nn.Module):
    def __init__(self, in_ch, h1, h2, h3, heads, dropout):
        super().__init__()
        self.conv1 = GATv2Conv(in_ch, h1, heads=heads, concat=True, dropout=dropout)
        self.gn1   = GraphNorm(h1*heads)
        self.conv2 = GATv2Conv(h1*heads, h2, heads=heads, concat=True, dropout=dropout)
        self.gn2   = GraphNorm(h2*heads)
        self.conv3 = GATv2Conv(h2*heads, h3, heads=heads, concat=False, dropout=dropout)
        self.gn3   = GraphNorm(h3)
        self.lin   = nn.Linear(h3, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.gn1(self.conv1(x, edge_index)))
        x = F.relu(self.gn2(self.conv2(x, edge_index)))
        x = F.relu(self.gn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)
        return self.lin(x)

class BCIPipeline:
    def __init__(self, method, fs=256):
        self.method = method.lower()
        self.fs     = fs

        if self.method == 'csp':
            filters   = _tr['filters']
            coef      = _tr['lda_coef']
            intercept = _tr['lda_intercept']
            self.csp = CSP(n_components=len(filters))
            self.csp.filters_ = filters
            self.lda = LinearDiscriminantAnalysis()
            self.lda.coef_      = coef
            self.lda.intercept_ = intercept

        elif self.method == 'plv':
            sd = torch.load(GAT_MODEL_PT, map_location='cpu')
            heads = sd['conv1.att'].shape[1]
            h1    = sd['conv1.att'].shape[2]
            h2    = sd['conv2.att'].shape[2]
            h3    = sd['conv3.att'].shape[2]
            in_ch = sd['conv1.lin_l.weight'].shape[1]
            self.gat = SimpleGAT(in_ch, h1, h2, h3, heads, dropout=0.1)
            self.gat.load_state_dict(sd)
            self.gat.eval()
        else:
            raise ValueError("Unknown method")

        self.adaptive = ADAPTATION
        self.adapt_N  = ADAPT_N
        self._win_buf = []
        self._lab_buf = []

        self.latest_plv = None  # for visualization

    def predict(self, window):
        if self.method == 'csp':
            arr  = window.T[np.newaxis,...]
            feat = self.csp.transform(arr)
            var  = np.var(feat, axis=2) if feat.ndim==3 else np.var(feat,axis=1,keepdims=True)
            vec  = np.log(var).ravel()
            return int(self.lda.predict([vec])[0])

        elif self.method == 'plv':
            adj = plvfcn(window)
            self.latest_plv = adj.copy()

            edge_index = threshold_graph_edges(adj, topk_percent=0.4)
            x = torch.eye(adj.shape[0], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index)
            out  = self.gat(data)
            return int(out.argmax(dim=1).item())

        else:
            cov = np.cov(window, rowvar=False)[np.newaxis,...]
            ts  = self.riemann_ts.transform(cov)
            return int(self.mdr.predict(ts)[0])

    def adapt(self):
        if not self.adaptive or len(self._win_buf) < self.adapt_N:
            return
        X = np.stack(self._win_buf)
        y = np.array(self._lab_buf)

        if self.method == 'csp':
            self.csp.fit(X, y)
            feats = np.log(np.var(self.csp.transform(X), axis=2))
            self.lda.fit(feats, y)

        elif self.method == 'plv':
            datas = []
            for w, lbl in zip(self._win_buf, self._lab_buf):
                adj = plvfcn(w)
                edge_index = threshold_graph_edges(adj, topk_percent=0.4)
                x = torch.eye(adj.shape[0], dtype=torch.float)
                d = Data(x=x, edge_index=edge_index, y=torch.tensor([lbl]))
                datas.append(d)

            loader = DataLoader(datas, batch_size=16, shuffle=True)
            opt = torch.optim.Adam(self.gat.parameters(), lr=1e-4)
            self.gat.train()
            for _ in range(5):
                for batch in loader:
                    opt.zero_grad()
                    out = self.gat(batch)
                    loss = F.cross_entropy(out, batch.y)
                    loss.backward()
                    opt.step()
            self.gat.eval()

        self._win_buf.clear()
        self._lab_buf.clear()
