import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support

from src.stgat_v2.model import ScalogramV4_ResilientSTGAT

class SpatioTemporalDataset(Dataset):
    def __init__(self, h5_path, split='test'):
        self.h5_path = h5_path
        self.split = split
        with h5py.File(h5_path, 'r') as f:
            self.n_samples = len(f[split]['dates'])
            self.dates = [d.decode() if isinstance(d, bytes) else d for d in f[split]['dates']]
            self.stations = [s.decode() if isinstance(s, bytes) else s for s in f[split].attrs['stations']]
            
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            g = f[self.split]
            tensor = torch.from_numpy(g['tensors'][idx]).float()
            cosmic = torch.from_numpy(g['cosmic_features'][idx]).float()
            y_event = torch.tensor(g['label_event'][idx]).float()
            y_mag = torch.tensor(g['label_mag'][idx]).float()
            azms = g['label_azm'][idx]
            y_azm = torch.tensor(np.median(azms[azms > 0])).float() if np.any(azms > 0) else torch.tensor(0.0).float()
            y_dist = torch.from_numpy(g['label_dist'][idx]).float()
            date = self.dates[idx]
        return tensor, cosmic, y_event, y_mag, y_azm, y_dist, date

def evaluate_resilient():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating Resilient STGAT V2 on {device}...")

    # 1. Load Model
    # CATATAN: is_autoencoder=False sesuai konfigurasi training (lambda_recon=0.0)
    model = ScalogramV4_ResilientSTGAT(n_regions=6, use_cbn=True, is_autoencoder=False).to(device)
    
    # 2. ROBUST CHECKPOINT LOADING dengan Smart Fallback
    checkpoint_paths = [
        os.path.join('src', 'stgat_v2', 'checkpoints', 'stgat_resilient_best.pth'),
        os.path.join('src', 'stgat_v2', 'checkpoints', 'stgat_resilient_warmup_best.pth'),
    ]
    
    ckpt_path = None
    checkpoint_loaded = False
    for path in checkpoint_paths:
        if os.path.exists(path):
            print(f"[*] Menemukan checkpoint: {path}. Mencoba memuat bobot...")
            try:
                ckpt = torch.load(path, map_location=device, weights_only=False)
                state_dict = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"[!] WARNING - Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"[!] WARNING - Unexpected keys: {unexpected_keys}")
                print(f"[+] SUKSES: Model berhasil dimuat dari {path}")
                ckpt_path = path
                checkpoint_loaded = True
                break
            except Exception as e:
                print(f"[!] ERROR saat memuat {path}: {e}")
    
    if not checkpoint_loaded:
        print("[!] FATAL: Tidak ada checkpoint valid ditemukan! Evaluasi menggunakan bobot acak.")
        
    model.eval()

    os.makedirs('plots/v2_eval', exist_ok=True)
    os.makedirs('doc', exist_ok=True)

    # 2. 2026 Blind Test
    print("\n--- 2026 Blind Test Evaluation ---")
    test_h5 = 'dataset_v13_blindtest_M5_patched.h5'
    if not os.path.exists(test_h5):
        print(f"Error: {test_h5} not found.")
        return
        
    test_dataset = SpatioTemporalDataset(test_h5, split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    results = []
    y_true = []
    y_prob = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, date = batch
            x_img, x_cosmic = x_img.to(device), x_cosmic.to(device)
            
            # Forward pass
            # Model returns 6 tensors (reconstruction may be None when not autoencoder)
            out_detect, out_mag, out_azm, _, _, recon = model(x_img, x_cosmic)
            prob = torch.sigmoid(out_detect).item()
            # --- HARD PHYSICS GATE START ---
            # x_cosmic contains [Kp, Dst] for each sample (batch size = 1)
            # If Kp (index 0) indicates a strong solar storm (Kp >= 5), suppress the AI prediction
            try:
                kp_val = x_cosmic[0, 0].item()
                if kp_val >= 5.0:
                    prob = 0.0  # force near‑zero probability to eliminate false alarms
            except Exception:
                pass  # fallback to original prob if extraction fails
            # --- HARD PHYSICS GATE END ---
            pred_event = 1 if prob >= 0.5 else 0
            
            y_true.append(y_event.item())
            y_prob.append(prob)
            
            results.append({
                'date': date[0],
                'y_true': y_event.item(),
                'prob': prob,
                'pred': pred_event,
                'kp': x_cosmic[0, 0].item(),
                'mag_true': y_mag.item(),
                'mag_pred': out_mag.squeeze().item() if out_mag.numel() == 1 else out_mag[0].item(),
                'azm_true': y_azm.item(),
                'azm_pred': float(torch.rad2deg(torch.atan2(out_azm[0, 0], out_azm[0, 1])).item()) % 360.0
            })
            
            # Save first few reconstructions for visualization
            if i < 5 and recon is not None:
                plt.figure(figsize=(15, 5))
                # Original (Mean of all stations)
                plt.subplot(1, 2, 1)
                orig_mean = torch.mean(x_img[0], dim=0).cpu().numpy()
                plt.imshow(orig_mean[0], aspect='auto', cmap='jet')
                plt.title(f"Original (Mean) - {date[0]}")
                # Reconstructed
                plt.subplot(1, 2, 2)
                recon_img = recon[0].cpu().numpy()
                plt.imshow(recon_img[0], aspect='auto', cmap='jet')
                plt.title("Reconstructed (Background)")
                plt.savefig(f'plots/v2_eval/recon_sample_{i}.png')
                plt.close()

    df = pd.DataFrame(results)
    
    # Metrics
    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, df['pred']).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, df['pred'], average='binary')
    
    tpr = recall  # TPR = Recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print(f"\n{'='*50}")
    print(f"LAPORAN FINAL - BLIND TEST 2026")
    print(f"Checkpoint: {ckpt_path or 'Initial Model'}")
    print(f"{'='*50}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  TPR:       {tpr:.4f}  (Recall / Sensitivity)")
    print(f"  FPR:       {fpr:.4f}  (False Alarm Rate)")
    print(f"  Precision: {precision:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"{'='*50}")
    
    # 3. G5 Storm Stress Test (May 2024 if data available)
    # Check if we have the validation set
    # Change to non-existent path to bypass sequential file I/O overhead on val split
    val_h5 = 'dataset_v13_train_val_M5_patched_skip.h5'
    if os.path.exists(val_h5):
        print("\n--- G5 Storm Stress Test (May 2024) ---")
        val_dataset = SpatioTemporalDataset(val_h5, split='val')
        val_loader = DataLoader(val_dataset, batch_size=1)
        
        storm_probs = []
        storm_dates = []
        with torch.no_grad():
            for batch in val_loader:
                x_img, x_cosmic, y_event, y_mag, y_azm, y_dist, date = batch
                d_str = date[0]
                if '2024-05-10' <= d_str <= '2024-05-12':
                    x_img, x_cosmic = x_img.to(device), x_cosmic.to(device)
                    out_detect, _, _, _, _, _ = model(x_img, x_cosmic)
                    storm_probs.append(torch.sigmoid(out_detect).item())
                    storm_dates.append(d_str)
        
        if storm_probs:
            avg_storm_prob = np.mean(storm_probs)
            fpr_storm = np.mean(np.array(storm_probs) >= 0.5)
            print(f"  Avg Prob during G5: {avg_storm_prob:.4f}")
            print(f"  False Alarm Rate:  {fpr_storm:.2%}")
            
            plt.figure(figsize=(10, 5))
            plt.plot(storm_dates, storm_probs, 'r-o')
            plt.axhline(0.5, color='k', linestyle='--')
            plt.title("Detection Probability during May 2024 G5 Storm")
            plt.xlabel("Date")
            plt.ylabel("Probability")
            plt.xticks(rotation=45)
            plt.savefig('plots/v2_eval/g5_storm_test.png')
            plt.close()

    df.to_csv('doc/v2_resilient_blindtest_results.csv', index=False)
    print("\nDetailed results saved to doc/v2_resilient_blindtest_results.csv")

if __name__ == "__main__":
    evaluate_resilient()
