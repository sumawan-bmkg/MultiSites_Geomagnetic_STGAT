# V2 to V7: Early Baselines
**Arsitektur Utama:** EfficientNet-B1 + BiGRU backbone. Fokus pada klasifikasi biner (Earthquake vs Noise).
**Hyperparameters:** LR 1e-4, Batch 32, Adam Optimizer.
**Hasil Metrik Final:** Val MAE Azimuth > 70 derajat.
**Insight:** Model mampu mendeteksi event tapi gagal dalam regresi arah (Azimuth) karena kurangnya informasi spasial.
