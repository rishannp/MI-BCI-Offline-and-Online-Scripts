import numpy as np
import pickle
import torch
import torch.nn as nn
from torch_geometric.data import Data
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
import scipy.signal as sig


def plvfcn(eegData: np.ndarray) -> np.ndarray:
    """
    Compute Phase Locking Value (PLV) matrix for EEG data.

    Args:
        eegData: array shape (n_times, n_channels)
    Returns:
        plvMatrix: array shape (n_channels, n_channels)
    """
    # Restrict to first 19 electrodes (adjust if needed)
    data = eegData[:, :16]
    n_times, n_channels = data.shape

    # Hilbert transform to get instantaneous phase
    analytic = sig.hilbert(data, axis=0)
    phase = np.angle(analytic)

    # Compute PLV for each pair
    plvMatrix = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase_diff = phase[:, j] - phase[:, i]
            plv = np.abs(np.sum(np.exp(1j * phase_diff)) / n_times)
            plvMatrix[i, j] = plv
            plvMatrix[j, i] = plv
    return plvMatrix


class DummyGAT(nn.Module):
    """
    A minimal GAT for testing: one GATConv layer + linear output
    """
    def __init__(self, in_feats: int, hid_feats: int, out_feats: int):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.conv1 = GATConv(in_feats, hid_feats, heads=1)
        self.lin = nn.Linear(hid_feats, out_feats)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index)
        h = torch.relu(h)
        # global mean pooling over nodes
        return self.lin(h).mean(0, keepdim=True)


class BCIPipeline:
    """
    Combined feature-extraction and classification pipeline.

    Supports methods:
      - 'csp': CSP + LDA
      - 'plv': PLV adjacency + GAT
      - 'riemann': Riemannian tangent-space + MDM
    """
    def __init__(self,
                 method: str,
                 fs: int = 256,
                 csp_path: str = None,
                 lda_path: str = None,
                 gat_path: str = None,
                 riemann_ts_path: str = None,
                 mdr_path: str = None):
        self.method = method.lower()
        self.fs = fs

        if self.method == 'csp':
            if not csp_path or not lda_path:
                raise ValueError("CSP pipeline requires csp_path and lda_path.")
            with open(csp_path, 'rb') as f:
                self.csp: CSP = pickle.load(f)
            with open(lda_path, 'rb') as f:
                self.lda: LinearDiscriminantAnalysis = pickle.load(f)

        elif self.method == 'plv':
            if not gat_path:
                raise ValueError("PLV pipeline requires gat_path.")
            self.gat = DummyGAT(in_feats=16, hid_feats=8, out_feats=2)
            state = torch.load(gat_path, map_location='cpu')
            self.gat.load_state_dict(state)
            self.gat.eval()

        elif self.method == 'riemann':
            if not riemann_ts_path or not mdr_path:
                raise ValueError("Riemannian pipeline requires riemann_ts_path and mdr_path.")
            with open(riemann_ts_path, 'rb') as f:
                self.riemann_ts: TangentSpace = pickle.load(f)
            with open(mdr_path, 'rb') as f:
                self.mdr: MDM = pickle.load(f)

        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'csp', 'plv', or 'riemann'.")

    def predict(self, window: np.ndarray) -> int:
        """
        Perform feature extraction and inference on a window.

        Args:
            window: np.ndarray of shape (n_samples, n_channels)
        Returns:
            int: predicted class label
        """
        if self.method == 'csp':
            # data shape: (1, n_channels, n_times)
            data = window.T[np.newaxis, ...]
            feat = self.csp.transform(data)
            # handle 2D or 3D output
            if feat.ndim == 3:
                var = np.var(feat, axis=2)
            else:
                var = np.var(feat, axis=1, keepdims=True)
            vec = np.log(var).ravel()
            return int(self.lda.predict([vec])[0])

        elif self.method == 'plv':
            adj = plvfcn(window)
            # build PyG Data object
            edge_idx = np.vstack(np.nonzero(adj))
            edge_index = torch.tensor(edge_idx, dtype=torch.long)
            edge_weight = torch.tensor(adj[adj != 0], dtype=torch.float)
            num_nodes = adj.shape[0]
            x = torch.eye(num_nodes, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
            with torch.no_grad():
                out = self.gat(data.x, data.edge_index, data.edge_weight)
            return int(out.argmax(dim=1).item())

        elif self.method == 'riemann':
            cov = np.cov(window, rowvar=False)
            ts = self.riemann_ts.transform([cov])
            return int(self.mdr.predict(ts)[0])

        else:
            raise RuntimeError("Invalid pipeline method.")
