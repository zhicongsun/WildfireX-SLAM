# Benchmarks (planned)

This directory will hold evaluation protocols, baseline configurations and reference results.

| Benchmark | Dataset | Splits | Status |
|---|---|---|---|
| WildfireX-SLAM v2.0 A, CS1–CS4 | WildfireX-SLAM v2.0 | `splits/wildfirex_slam_v2_0.json` in the dataset | data released; evaluation scripts planned |

Planned metrics: ATE RMSE (tracking), PSNR / SSIM / LPIPS (rendering), depth L1 / RMSE (geometry),
reported per sequence and per challenge factor. Baselines: SplaTAM, MonoGS, Co-SLAM, ORB-SLAM3.
