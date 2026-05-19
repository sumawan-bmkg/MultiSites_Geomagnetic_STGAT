# V8: SupCon & FPR Suppression
**Arsitektur Utama:** Supervised Contrastive Learning (SupCon). Memaksa manifold embedding positif dan negatif terpisah secara ekstrem.
**Hyperparameters:** Temperature 0.07, Label Smoothing 0.1.
**Hasil Metrik Final:** FPR 0.125, Recall 0.972, EWS Score +0.829.
**Insight:** Berhasil menekan False Positive akibat badai matahari, namun terjadi 'Mode Collapse' pada prediksi azimuth (prediksi mengumpul di satu titik).
