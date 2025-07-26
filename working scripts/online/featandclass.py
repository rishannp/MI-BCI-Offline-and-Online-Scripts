# featandclass.py

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool
from torch_geometric.utils import add_self_loops
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
import scipy.signal as sig

from config import (
    METHOD, ADAPTATION, ADAPT_N,
    TRAINING_DATA, GAT_MODEL_PT,
    SUBJECT_DIR, SESSION_DIR
)

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
    plv = plv.copy()
    np.fill_diagonal(plv, 0.0)
    triu = np.triu_indices(plv.shape[0], k=1)
    weights = plv[triu]
    k = max(1, int(len(weights) * topk_percent))
    idx = np.argpartition(weights, -k)[-k:]
    row, col = triu[0][idx], triu[1][idx]
    edge_index = np.hstack([np.stack([row,col]), np.stack([col,row])])
    edge_index, _ = add_self_loops(
        torch.tensor(edge_index, dtype=torch.long),
        num_nodes=plv.shape[0]
    )
    return edge_index

class SimpleGAT(nn.Module):
    def __init__(self, in_ch, h1, h2, h3, heads, dropout=0.1):
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
    def __init__(self, method=METHOD, fs=256):
        self.method = method.lower()
        self.fs     = fs

        # load static model
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
            h1, h2, h3 = sd['conv1.att'].shape[2], sd['conv2.att'].shape[2], sd['conv3.att'].shape[2]
            in_ch = sd['conv1.lin_l.weight'].shape[1]
            self.gat = SimpleGAT(in_ch, h1, h2, h3, heads)
            self.gat.load_state_dict(sd)
            self.gat.eval()
        else:
            raise ValueError(f"Unknown method {self.method!r}")

        self.adaptive = ADAPTATION
        self.adapt_N  = ADAPT_N
        self._win_buf = []
        self._lab_buf = []
        self.latest_plv = None

    def predict(self, window):
        if self.method == 'csp':
            arr = window.T[np.newaxis,...]
            feat = self.csp.transform(arr)
            vec = np.log(np.var(feat, axis=(1,2))).ravel()
            return int(self.lda.predict([vec])[0])

        # PLV path
        adj = plvfcn(window)
        self.latest_plv = adj.copy()
        if self.method == 'plv':
            ei = threshold_graph_edges(adj, topk_percent=0.4)
            x  = torch.eye(adj.shape[0])
            data = Data(x=x, edge_index=ei)
            out = self.gat(data)
            return int(out.argmax(dim=1).item())

        # fallback
        raise RuntimeError("Invalid method")

    def adapt(self):
        if not self.adaptive or len(self._win_buf) < self.adapt_N:
            return
        X = np.stack(self._win_buf)
        y = np.array(self._lab_buf)

        if self.method == 'csp':
            self.csp.fit(X, y)
            feats = np.log(np.var(self.csp.transform(X), axis=2))
            self.lda.fit(feats, y)
            # save updated CSP+LDA
            with open(os.path.join(SESSION_DIR, "csp_lda_updated.pkl"), 'wb') as f:
                pickle.dump({'filters':self.csp.filters_,
                             'lda_coef':self.lda.coef_,
                             'lda_intercept':self.lda.intercept_},
                            f)

        else:  # plv
            datas = []
            for w, lbl in zip(self._win_buf, self._lab_buf):
                adj = plvfcn(w)
                ei  = threshold_graph_edges(adj, topk_percent=0.4)
                x   = torch.eye(adj.shape[0])
                datas.append(Data(x=x, edge_index=ei, y=torch.tensor([lbl])))
            loader = DataLoader(datas, batch_size=16, shuffle=True)
            opt = torch.optim.Adam(self.gat.parameters(), lr=1e-4)
            self.gat.train()
            for _ in range(5): # NUMBER OF EPOCHS for Adaptation
                for batch in loader:
                    opt.zero_grad()
                    out = self.gat(batch)
                    loss = F.cross_entropy(out, batch.y)
                    loss.backward()
                    opt.step()
            self.gat.eval()
            # save updated GAT
            torch.save(self.gat.state_dict(), os.path.join(SESSION_DIR, "gat_updated.pt"))

        self._win_buf.clear()
        self._lab_buf.clear()
