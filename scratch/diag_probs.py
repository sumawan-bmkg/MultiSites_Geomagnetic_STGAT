import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class SpatialGNNModule(nn.Module):
    def __init__(self, in_features=512, hidden=256, out_features=512):
        super().__init__()
        self.gat1 = nn.Linear(in_features, hidden)
        self.gat2 = nn.Linear(hidden, out_features)
        self.attention = nn.Parameter(torch.ones(24))
    def forward(self, x):
        h = F.relu(self.gat1(x))
        h = self.gat2(h)
        weights = F.softmax(self.attention, dim=0)
        consensus = torch.sum(h * weights.view(1, 24, 1), dim=1)
        return consensus, weights

class ScalogramV4_STGAT(nn.Module):
    def __init__(self, n_regions=6):
        super().__init__()
        efficientnet = models.efficientnet_b1(weights=None)
        self.spatial_encoder = efficientnet.features
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.temporal_encoder = nn.GRU(
            input_size=1280, hidden_size=256, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.2
        )
        self.gat_layer = SpatialGNNModule(in_features=512, hidden=256, out_features=512)
        self.cosmic_gate = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 512), nn.Sigmoid()
        )
        self.alpha_regions = nn.Parameter(torch.randn(n_regions))
        self.head_detect = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        self.head_mag = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.head_azm = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x_img, x_cosmic):
        B, N, C, H, W = x_img.shape
        x = x_img.view(B * N, C, H, W)
        feat = self.spatial_encoder(x)
        feat = self.pool(feat).squeeze(2).permute(0, 2, 1)
        self.temporal_encoder.flatten_parameters()
        gru_out, _ = self.temporal_encoder(feat)
        v_img = torch.mean(gru_out, dim=1)
        v_img = v_img.view(B, N, 512)
        v_consensus, att_weights = self.gat_layer(v_img)
        gate = self.cosmic_gate(x_cosmic)
        v_fusion = v_consensus * gate
        out_detect = self.head_detect(v_fusion).squeeze(-1)
        out_mag = self.head_mag(v_fusion).squeeze(-1)
        out_azm = self.head_azm(v_fusion)
        return out_detect, out_mag, out_azm, self.alpha_regions

class SpatioTemporalDataset(Dataset):
    def __init__(self, h5_path, split='train'):
        self.h5_path = h5_path
        self.split = split
        with h5py.File(h5_path, 'r') as f:
            self.n_samples = len(f[split]['dates'])
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            g = f[self.split]
            tensor = torch.from_numpy(g['tensors'][idx]).float()
            cosmic = torch.from_numpy(g['cosmic_features'][idx]).float()
            y_event = torch.tensor(g['label_event'][idx]).float()
        return tensor, cosmic, y_event

def diag():
    device = 'cpu'
    model = ScalogramV4_STGAT(n_regions=6).to(device)
    model.load_state_dict(torch.load('checkpoints/best_stgat_v1.pth', map_location=device))
    model.eval()

    test_dataset = SpatioTemporalDataset('dataset_v13_blindtest_M5_patched.h5', split='test')
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    probs = []
    events = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x_img, x_cosmic, y_event = batch
            out_detect, _, _, _ = model(x_img.to(device), x_cosmic.to(device))
            p = torch.sigmoid(out_detect).item()
            probs.append(p)
            events.append(y_event.item())
            if i >= 40: break
            
    probs = np.array(probs)
    events = np.array(events)
    
    print(f"Diagnostic on first 40 blind samples:")
    print(f"Max Prob: {np.max(probs):.6f}")
    print(f"Min Prob: {np.min(probs):.6f}")
    print(f"Mean Prob: {np.mean(probs):.6f}")
    
    event_probs = probs[events == 1]
    noise_probs = probs[events == 0]
    print(f"Avg Prob on Events: {np.mean(event_probs):.6f} (N={len(event_probs)})")
    print(f"Avg Prob on Noise: {np.mean(noise_probs):.6f} (N={len(noise_probs)})")

if __name__ == "__main__":
    diag()
