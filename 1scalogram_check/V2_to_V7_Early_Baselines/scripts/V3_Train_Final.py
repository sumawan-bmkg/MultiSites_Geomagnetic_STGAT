import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from pathlib import Path
import argparse
import psutil

# Local Imports
from V3_Model import MultiTaskScalogramV3
from V3_HDF5_DataLoader import create_v3_dataloaders

# ---------------------------------------------------------
# 1. Specialized Loss Functions
# ---------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        log_pt = -self.ce(inputs, targets)
        pt = torch.exp(log_pt)
        loss = -self.alpha * (1 - pt) ** self.gamma * log_pt
        return loss

class VonMisesLoss(nn.Module):
    def __init__(self, kappa=1.0):
        super(VonMisesLoss, self).__init__()
        self.kappa = kappa

    def forward(self, pred, target_rad, reduce=True):
        # pred: (B, 2) -> [sin, cos]
        # target_rad: (B) in radians
        pred_sin, pred_cos = pred[:, 0], pred[:, 1]
        target_sin, target_cos = torch.sin(target_rad), torch.cos(target_rad)
        
        # Mean Squared Error on the vector components
        loss = (pred_sin - target_sin)**2 + (pred_cos - target_cos)**2
        
        if reduce:
            return torch.mean(loss)
        return loss

# ---------------------------------------------------------
# 2. Main Training Engine
# ---------------------------------------------------------
def run_training_v3(data_h5, epochs=15, batch_size=16, lr=1e-4):
    print("="*60)
    print(" SCALOGRAM V3: SENSOR FUSION TRAINING (STABLE ENGINE) ")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model
    print(">>> Initializing Architecture...")
    model = MultiTaskScalogramV3(pretrained=False).to(device)
    
    # Optimizer & Losses
    # weight_decay dinaikkan ke 1e-2 (User request)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    start_epoch = 0
    start_batch = 0
    best_val_loss = float('inf')

    # Data
    print(f">>> Loading HDF5 Dataset: {os.path.basename(data_h5)}")
    train_loader, val_loader = create_v3_dataloaders(data_h5, batch_size=batch_size, num_workers=0)
    criterion_det = FocalLoss(alpha=1.0, gamma=2.0)
    criterion_mag = nn.CrossEntropyLoss()
    criterion_azm = VonMisesLoss()
    
    # MIXED PRECISION (AMP) for 2GB VRAM MX250
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    
    # TASK 2: LOGIKA PEMULIHAN (LOAD/RESUME CHECKPOINT)
    checkpoint_path_intra = 'checkpoints/v3_intra_epoch_checkpoint.pth'
    checkpoint_path_epoch = 'checkpoints/v3_latest_checkpoint.pth'
    
    ckpt_path_to_load = None
    if os.path.exists(checkpoint_path_intra):
        ckpt_path_to_load = checkpoint_path_intra
    elif os.path.exists(checkpoint_path_epoch):
        ckpt_path_to_load = checkpoint_path_epoch
        
    if ckpt_path_to_load:
        print(f">>> Found existing checkpoint: {ckpt_path_to_load}. Resuming...")
        # CRITICAL FIX for MX250 2GB VRAM OOM: Map strictly to 'cpu' during unpacking
        ckpt = torch.load(ckpt_path_to_load, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            
        if 'batch_idx' in ckpt:
            start_epoch = ckpt['epoch']
            start_batch = ckpt['batch_idx'] + 1
            best_val_loss = ckpt.get('best_loss', float('inf'))
            print(f"[INFO] Resuming training from Epoch {start_epoch + 1}, Batch {start_batch}...")
        else:
            start_epoch = ckpt['epoch'] + 1
            start_batch = 0
            best_val_loss = ckpt.get('best_loss', float('inf'))
            print(f"[INFO] Resuming training from Epoch {start_epoch + 1}...")
            
        del ckpt
        torch.cuda.empty_cache()

    
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss_accum = 0
        epoch_start_time = time.time()
        
        print(f"\n[Epoch {epoch+1}/{epochs}]")
        torch.cuda.empty_cache()
        
        for i, (img, cosmic, y_det, y_mag_raw, y_azm) in enumerate(train_loader):
            img, cosmic = img.to(device), cosmic.to(device)
            y_det, y_azm = y_det.long().to(device), y_azm.to(device)
            
            # Stage Mapping (Multi-Class 0-3 based on EWS Logic)
            y_mag = torch.zeros_like(y_mag_raw, dtype=torch.long)
            for j, mag in enumerate(y_mag_raw):
                if mag == 0.0: y_mag[j] = 0
                elif mag < 5.0: y_mag[j] = 1 # Stage 1
                elif mag < 6.0: y_mag[j] = 2 # Stage 2
                else: y_mag[j] = 3           # Stage 3
            y_mag = y_mag.to(device)
            
            optimizer.zero_grad()
            
            # Forward with AMP
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                out_det, out_mag, out_azm = model(img, cosmic)
                
                loss_det = criterion_det(out_det, y_det)
                loss_mag = criterion_mag(out_mag, y_mag)
                
                y_azm_rad = y_azm * (torch.pi / 180.0)
                loss_azm_raw = criterion_azm(out_azm, y_azm_rad, reduce=False)
                mask_anomali = (y_det == 1).float()
                loss_azm = (loss_azm_raw * mask_anomali).sum() / (mask_anomali.sum() + 1e-8)
                
                total_loss = (1.0 * loss_det) + (1.0 * loss_mag) + (5.0 * loss_azm)
            
            # Backward with AMP
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss_accum += total_loss.item()
            
            # EXPLICIT CPU RAM FREEING (Anti CPU-Leak Patch)
            del img, cosmic, y_det, y_mag_raw, y_mag, y_azm, y_azm_rad
            del out_det, out_mag, out_azm, loss_det, loss_mag, loss_azm_raw, mask_anomali, loss_azm, total_loss
            
            if i % 20 == 0:
                print(f"  Batch {i}/{len(train_loader)} | Loss: {total_loss_accum/(i+1 - start_batch):.4f}", flush=True)
                import gc
                gc.collect()
                
            if i % 50 == 0:
                ram_pct = psutil.virtual_memory().percent
                print(f"  [RAM MONITOR] Batch {i} | System RAM: {ram_pct}%", flush=True)
                
            if i % 100 == 0 and i > 0:
                intra_path = 'checkpoints/v3_intra_epoch_checkpoint.pth'
                torch.save({
                    'epoch': epoch,
                    'batch_idx': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_loss': best_val_loss,
                }, intra_path)
                print(f"  (!) Intra-Epoch Checkpoint Saved: Batch {i}")
        
        avg_train_loss = total_loss_accum / (len(train_loader) - start_batch)
        duration = time.time() - epoch_start_time
        print(f"  Summary: Train Loss {avg_train_loss:.5f} | Time: {duration:.1f}s")
        
        # Advanced Validation
        model.eval()
        val_loss_accum = 0
        tp, fp, tn, fn = 0, 0, 0, 0
        
        with torch.no_grad():
            for img, cosmic, y_det, y_mag_raw, y_azm in val_loader:
                img, cosmic = img.to(device), cosmic.to(device)
                y_det, y_azm = y_det.long().to(device), y_azm.to(device)
                
                # Stage Mapping (Multi-Class 0-3 based on EWS Logic)
                y_mag = torch.zeros_like(y_mag_raw, dtype=torch.long)
                for j, mag in enumerate(y_mag_raw):
                    if mag == 0.0: y_mag[j] = 0
                    elif mag < 5.0: y_mag[j] = 1 # Stage 1
                    elif mag < 6.0: y_mag[j] = 2 # Stage 2
                    else: y_mag[j] = 3           # Stage 3
                y_mag = y_mag.to(device)
                
                out_det, out_mag, out_azm = model(img, cosmic)
                
                # Metrics for Detection
                preds = torch.argmax(out_det, dim=1)
                for p, t in zip(preds, y_det):
                    if p == 1 and t == 1: tp += 1
                    elif p == 1 and t == 0: fp += 1
                    elif p == 0 and t == 0: tn += 1
                    elif p == 0 and t == 1: fn += 1
                
                loss_det = criterion_det(out_det, y_det)
                loss_mag = criterion_mag(out_mag, y_mag)
                
                # --- SAFEGUARD 2 & 3: VALIDATION MASKING & SCALING ---
                y_azm_rad = y_azm * (torch.pi / 180.0)
                loss_azm_raw = criterion_azm(out_azm, y_azm_rad, reduce=False)
                mask_anomali = (y_det == 1).float()
                loss_azm = (loss_azm_raw * mask_anomali).sum() / (mask_anomali.sum() + 1e-8)
                
                val_loss_accum += (1.0 * loss_det + 1.0 * loss_mag + 5.0 * loss_azm).item()
        
        avg_val_loss = val_loss_accum / len(val_loader)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"  Validation Loss: {avg_val_loss:.5f}")
        print(f"  Metrics: Prec={precision:.3f} | Rec={recall:.3f} | FPR={fpr:.4f} (FP: {fp})")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('checkpoints', exist_ok=True)
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_val_loss,
                'metrics': {'prec': precision, 'rec': recall, 'fpr': fpr}
            }
            torch.save(state, 'checkpoints/v3_best_val_checkpoint.pth')
            torch.save(model.state_dict(), 'checkpoints/v3_fusion_best.pth')
            print("  (!) Best Model Saved.")
        
        # TASK 1: LOGIKA PENYIMPANAN (SAVE CHECKPOINT)
        latest_path = 'checkpoints/v3_latest_checkpoint.pth'
        latest_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_loss': best_val_loss,
        }
        torch.save(latest_state, latest_path)
        print(f"  (!) Latest Checkpoint Saved: Epoch {epoch+1}")
        
        # Clean up intra-epoch checkpoint properly
        if os.path.exists('checkpoints/v3_intra_epoch_checkpoint.pth'):
            os.remove('checkpoints/v3_intra_epoch_checkpoint.pth')
            
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ScalogramV3 Training Engine (Stable)")
    parser.add_argument('--data_path', type=str, default=r"d:\multi\scalogramv3\scalogram_v3_cosmic.h5")
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    if os.path.exists(args.data_path):
        run_training_v3(args.data_path, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    else:
        print(f"Error: {args.data_path} not found.")
