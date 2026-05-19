import torch
import sys
import os
import h5py
import numpy as np

# Add project root to path to allow absolute imports from src
_SCRATCH_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRATCH_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.stgat_v2.model import ScalogramV4_ResilientSTGAT
from src.stgat_v2.losses import PhysicsResilientLoss


def test():
    device = torch.device('cpu')
    print(f"Testing on {device}")
    
    h5_path = 'dataset_v13_train_val_M5_patched.h5'
    if not os.path.exists(h5_path):
        print(f"H5 file not found at {h5_path}")
        return

    print("Initializing model...")
    model = ScalogramV4_ResilientSTGAT(n_regions=6, use_cbn=True, is_autoencoder=False).to(device)
    
    print("Freezing spatial_encoder...")
    for param in model.spatial_encoder.parameters():
        param.requires_grad = False
        
    criterion = PhysicsResilientLoss(lambda_recon=0.0).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    print("Loading one sample...")
    with h5py.File(h5_path, 'r') as f:
        g = f['train']
        x_img = torch.tensor(g['tensors'][0:1]).float().to(device)
        x_cosmic = torch.tensor(g['cosmic_features'][0:1]).float().to(device)
        y_event = torch.tensor(g['label_event'][0:1]).float().to(device)
        y_mag = torch.tensor(g['label_mag'][0:1]).float().to(device)
        y_azm = torch.tensor([0.0]).float().to(device)
        y_dist = torch.tensor(g['label_dist'][0:1]).float().to(device)
        region_idx = torch.tensor([0]).to(device)

    print("Running forward pass (Sequential Stations)...")
    optimizer.zero_grad()
    
    B, N, C, H, W = x_img.shape
    phys_embed = model.physics_sidecar(x_cosmic)
    station_feats = []
    
    try:
        for i in range(N):
            print(f"  Processing station {i+1}/{N}...", flush=True)
            x_station = x_img[:, i, :, :, :]
            for m in model.cbn_layers:
                m.current_embedding = phys_embed
            
            feat_station = model.spatial_encoder(x_station)
            feat_station = model.pool(feat_station).squeeze(2).permute(0, 2, 1)
            station_feats.append(feat_station)
            
        print("\nStations processed. Combining...")
        feat = torch.cat(station_feats, dim=0)
        
        # Manually complete the forward pass
        model.temporal_encoder.flatten_parameters()
        gru_out, _ = model.temporal_encoder(feat)
        v_img = torch.mean(gru_out, dim=1) # (B*N, 512)
        v_img = v_img.view(B, N, 512)
        v_consensus, att_weights = model.gat_layer(v_img)
        
        raw_pred_event = model.head_detect(v_consensus).squeeze(-1)
        gate_val = model.cosmic_gate(x_cosmic)
        v_fusion = v_consensus * gate_val
        out_mag = model.head_mag(v_fusion).squeeze(-1)
        out_azm = model.head_azm(v_fusion)
        
        preds = (raw_pred_event, out_mag, out_azm, model.alpha_regions, gate_val, None)
        
        print("Calculating loss...")
        targets = (y_event, y_mag, y_azm, y_dist, region_idx, x_img)
        loss, metrics = criterion(preds, targets)
        print(f"Loss: {loss.item()}")

        print("Running backward pass...")
        loss.backward()
        print("Backward pass successful!")
        
        optimizer.step()
        print("Step successful!")
        
    except Exception as e:
        print(f"\nCRASHED at station {len(station_feats)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
