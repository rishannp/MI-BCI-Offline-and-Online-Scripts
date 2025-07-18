# generate_dummy_models.py

import os
import pickle
import numpy as np
import torch
import torch.nn as nn

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from torch_geometric.nn import GATConv

# 1) CSP + LDA
# ---------------
# Fake epochs: 10 trials × 16 channels × 256 time-points
fake_epochs = np.random.randn(10, 16, 256)
fake_labels = np.random.randint(0, 2, size=(10,))

csp = CSP(n_components=4)
csp.fit(fake_epochs, fake_labels)

# Transform and compute log-variance, handling both 2D & 3D outputs
transformed = csp.transform(fake_epochs)
if transformed.ndim == 3:
    # shape (n_epochs, n_components, n_times)
    feats = np.log(np.var(transformed, axis=2))
elif transformed.ndim == 2:
    # shape (n_epochs, n_components)
    feats = np.log(np.var(transformed, axis=1, keepdims=True))
else:
    raise RuntimeError(f"Unexpected CSP output ndim={transformed.ndim}")

# Fit LDA
lda = LinearDiscriminantAnalysis()
lda.fit(feats, fake_labels)

with open("csp.pkl", "wb") as f:
    pickle.dump(csp, f)
with open("lda.pkl", "wb") as f:
    pickle.dump(lda, f)

# 2) PLV + DummyGAT
class DummyGAT(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = GATConv(in_feats, hid_feats, heads=1)
        self.lin   = nn.Linear(hid_feats, out_feats)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index)
        h = torch.relu(h)
        return self.lin(h).mean(0, keepdim=True)

# Instantiate and SAVE only the state_dict
gat = DummyGAT(in_feats=16, hid_feats=8, out_feats=2)
torch.save(gat.state_dict(), "gat_model.pth")   # <<-- state_dict, not entire module


# 3) Riemannian + MDM
# --------------------
# Fake covariance matrices (10 samples of 16×16 SPD)
covs = []
for _ in range(10):
    A = np.random.randn(16, 16)
    covs.append((A @ A.T) + np.eye(16)*1e-6)
covs = np.stack(covs, axis=0)
labels = np.random.randint(0, 2, size=(10,))

ts = TangentSpace().fit(covs, labels)
mdr = MDM().fit(covs, labels)

with open("riemann_ts.pkl", "wb") as f:
    pickle.dump(ts, f)
with open("mdr_model.pkl", "wb") as f:
    pickle.dump(mdr, f)


print("Dummy models saved:")
print("  • csp.pkl")
print("  • lda.pkl")
print("  • gat_model.pth")
print("  • riemann_ts.pkl")
print("  • mdr_model.pkl")
