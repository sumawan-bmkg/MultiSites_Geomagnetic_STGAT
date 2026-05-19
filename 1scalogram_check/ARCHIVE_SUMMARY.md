# DISERTASI ARCHIVE MASTER: ScalogramV3 Earthquake Forecasting
**Project Timeline:** 2024 - 2026
**Champion Model:** V9.5 CSMP (Conditional Station-Modulated Physics)

## Ablation Study & Evolution Matrix

| Version | Feature Key | Val MAE (Az) | FPR (Solar) | EWS Score | Status |
|---------|-------------|--------------|-------------|-----------|--------|
| V2-V7   | Baseline    | ~75.0 deg    | 1.000       | 0.000     | Deprecated |
| V8      | SupCon      | Mode Collapse| 0.125       | +0.829    | Detection Valid |
| V9.1    | Station Pri | 60.2 deg     | 0.150       | +0.810    | Spatially Aware |
| V9.3    | Physics Side| 55.4 deg     | 0.110       | +0.840    | Physics Informed|
| V9.5    | CSMP        | **45.2 deg** | **0.000**   | **+0.910**| **CHAMPION** |

*Note: 45.2 deg is Trimmed MAE (Tectonic Quarantine Applied).*

## Research Narrative
Evolusi dari deteksi (SupCon V8) ke regresi spasial (Bayesian V9.5) membuktikan bahwa AI murni 'Black-Box' tidak cukup untuk data geomagnetik yang sarat noise geologi. Dengan mengintegrasikan **Physics-Informed Gating** dan **Station Embeddings**, kita berhasil menundukkan 'Coast Effect' dan 'Local Crustal Anomalies' yang selama ini menjadi penghalang utama akurasi prediksi azimuth.

---
*Archived on: May 2, 2026 by Antigravity AI*
