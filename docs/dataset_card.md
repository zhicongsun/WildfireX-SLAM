---
pretty_name: WildfireX-SLAM
license: other
license_name: wildfirex-slam-data-license
license_link: LICENSE.md
task_categories: [depth-estimation, image-to-3d]
tags: [slam, rgb-d, multi-modal, wildfire, forest, uav, gaussian-splatting, synthetic, simulation]
size_categories: [10K<n<100K]
---
<!-- Source of the Hugging Face dataset card; tools/stage_dataset.py copies it to README.md next to LICENSE.md. -->

# WildfireX-SLAM v2.0

Synthetic low-altitude UAV sequences over a forest with active fires — RGB, depth, normals, materials and exact poses —
for SLAM, 3D Gaussian Splatting and robust perception under smoke, flames, falling leaves, fog and changing illumination.

v2.0 is a re-collection of the WildfireX-SLAM benchmark structure (v1.0, MMM 2026 paper) with the open `wildfirex`
toolkit on simulator WildfireX-Sim 1.1: six fire sites,
canopy-following flights just above the trees, float depth, exact camera poses, 30 Hz with 0.1 m per frame,
runtime-controlled conditions and automatic quality checks. Every sequence can be regenerated with
`wildfirex collect wildfirex_slam_v2.yaml --only <id>` (toolkit: `pip install wildfirex`; code, config and docs in the
[WildfireX-SLAM repository](https://github.com/zhicongsun/WildfireX-SLAM); project page:
[zhicongsun.github.io/wildfirexslam](https://zhicongsun.github.io/wildfirexslam)).

| | |
|---|---|
| Sequences | 19 (`sequences/<site>-<number>`, sites `ed01` … `ed06`) |
| Frames | 16,572 RGB-D frames at 640×480, 30 Hz |
| Modalities | RGB, depth (cm), normal, albedo, roughness, specular, AirSim object-ID "infrared" |
| Ground truth | camera-to-world poses (TUM + Replica formats), intrinsics, velocity/acceleration, fire-kit positions |
| Flight altitude | follows the surveyed canopy with a minimum clearance of 12 m (spirals of A) or 15 m (all others); per-sequence median 29–42 m above ground, 17–24 m above the canopy |
| Simulator | WildfireX-Sim 1.1 (Electric Dreams, UE 5.2, AirSim-WildfireX plugin; + movable fire kits, moon) |
| Collected with | wildfirex 0.1.0 (`wildfirex_slam_v2.yaml`) |
| Size | ≈ 58 GB (PNG) |

## Benchmarks (`splits/wildfirex_slam_v2_0.json`)

| Benchmark | Sequences | Varied factor | Trajectory |
|---|---|---|---|
| A | ed01-0001 … 0004, ed02-0001 … 0004 | no fire vs. fire, two sites | T01 spiral around the fire (42 m away), T02 lawn-mower survey 55 m beside the fire looking at it |
| CS1 | ed03-0001 … 0004 | falling leaves | T03 search flight towards the fire (ends 45 m from it), T04 spiral |
| CS2 | ed04-0001 … 0004 | morning / noon / dusk / night | T04 spiral |
| CS3 | ed05-0001, 0002 | fog 0 vs 0.7 | T02 side survey |
| CS4 | ed06-0001 | fire + leaves + fog 0.35 + night | T04 spiral |

Paired sequences share exactly the same trajectory and fire. Per-sequence conditions are in `sequences.csv`
and each `meta.json`.

Fire: three fire kits (ground flames + smoke) form a ≈ 24 m fire front at every site; at `ed01` the level's original
fire and smoke column (as in v1.0) are kept as well. At dusk a low sun lights the canopy from the opposite side to morning and the fire
glows faintly; at night a weak moon and the fire light the scene, so it is dark away from the fire.

## Download

Each sequence folder holds its poses and metadata as small text / json files plus one zip per modality
(`rgb.zip`, `depth.zip`, `normal.zip`, `albedo.zip`, `roughness.zip`, `specular.zip`, `ir.zip`), so you can fetch only
what you need. RGB + depth for all sequences is ≈ 17.5 GB of the ≈ 62 GB total.

```bash
pip install -U huggingface_hub
# RGB-D, poses and metadata of all 19 sequences (~17.5 GB)
huggingface-cli download zhicongsun/WildfireX-SLAM --repo-type dataset --local-dir WildfireX-SLAM   --include "*.md" "*.csv" "*.yaml" "splits/*" "survey/*" "sequences/*/*.txt" "sequences/*/*.json"             "sequences/*/*.yaml" "sequences/*/*.csv" "sequences/*/rgb.zip" "sequences/*/depth.zip"
# one complete sequence
huggingface-cli download zhicongsun/WildfireX-SLAM --repo-type dataset --local-dir WildfireX-SLAM --include "sequences/ed01-0002/*"
# everything (~62 GB)
huggingface-cli download zhicongsun/WildfireX-SLAM --repo-type dataset --local-dir WildfireX-SLAM
```

Unzip in place (each `<modality>.zip` extracts to `<modality>/000000.png ...` inside its sequence folder):

```bash
python -c "import glob,os,zipfile; [zipfile.ZipFile(z).extractall(os.path.dirname(z)) for z in glob.glob('WildfireX-SLAM/sequences/*/*.zip')]"
```

## Format

See [`docs/data_format.md`](https://github.com/zhicongsun/WildfireX-SLAM/blob/main/docs/data_format.md) in the
WildfireX-SLAM repository. In short: `depth/*.png` are uint16 centimetres (0 = invalid); poses are OpenCV
camera-to-world with the world in AirSim NED (metres); `intrinsics.yaml` gives fx = fy = 320, cx = 320, cy = 240.
`meta.json` records `ned_origin_ue_cm` (the AirSim NED origin in Unreal world coordinates), the fire-kit
positions and all versions, so sequences from different sites can be put in one world frame.

### Sequence ids

`<site>-<number>`, e.g. `ed04-0003`: `ed` = the Electric Dreams map, `01`–`06` = fire site (`ed01` is the paper's
fire, `ed02`–`ed06` were chosen in the forest from the canopy survey), `0001`… = running number at that site. Conditions
(fire, light, fog, leaves, trajectory) are fields in `sequences.csv` and `meta.json`, never part of the id, so new
conditions never rename data.

### Modalities (all 640×480, same frame)

| folder | content |
|---|---|
| `rgb` | final rendered colour image |
| `depth` | planar depth (distance along the optical axis), uint16 PNG in cm, 0 = invalid / > 655 m; includes smoke |
| `normal` | surface normal from the G-buffer, encoded as RGB |
| `albedo` | base colour of the surface (material colour without lighting, shadows or fog) from the G-buffer |
| `roughness` | material roughness (G-buffer), grey |
| `specular` | material specular (G-buffer), grey |
| `ir` | AirSim object-ID "infrared" image (grey value per object), **not** a physical thermal image |

## Differences from v1.0 (paper) — please read

- Coverage: six fire sites within ≈ 600 m of the central forest, each flown over ≈ 100–150 m — not the 16 km² of v1.0.
- Altitude: canopy-following, at least 12–15 m above the canopy, 26–53 m above ground, median ≤ 42 m (v1.0: "< 50 m above ground", fixed height).
- Views: the camera looks at the fire from 42–110 m (spiral, side survey, search); flying right over a fire puts the camera
  inside the smoke column (> 50 % of the image), so v2.0 has no nadir pass over the flames.
- No physical thermal images (`ir` is AirSim's object-ID image); no near-ground views.
- Depth includes smoke, flames and falling leaves (same instant as RGB); foliage moves with wind.
- Fire sites `ed02`–`ed06` use fire kits added to the level (WildfireX-Sim 1.1); `ed01` is the paper's fire.

## Quality

All 19 sequences pass the automatic checks (`qa.json`, `sequences.csv`): no missing frames, valid depth 96.7–100 %,
no camera-inside-geometry frames (no frame with > 20 % of pixels closer than 0.5 m), median relative depth reprojection
error between consecutive frames 0.08–0.42 % (higher where smoke, which is part of the depth, moves), and a consistent
first frame (5 pre-roll captures are dropped before frame 0; the share of inconsistent pixels between frames 0 and 1
matches that of later frame pairs).

Note: fog 0.7 (CS3) is dense; at night (CS4) the fire light makes fog glow, so CS4 uses fog 0.35.

## License

[WildfireX-SLAM Dataset License](LICENSE.md): the terms of **CC BY 4.0** (any use, including commercial, with
attribution) **plus one restriction — no use in, for the development of, or as training input to generative AI
programs.** The restriction passes on the "NoAI" terms of the third-party Unreal Engine assets depicted in the data
(Electric Dreams Environment by Epic Games, with its Quixel Megascans assets; M5 VFX Vol2 Fire and Flames). SLAM, pose and
depth estimation, 3D reconstruction and rendering of the captured scenes, detection, segmentation and benchmarking are
not affected. No rights to the underlying assets are granted.

Unreal® is a trademark or registered trademark of Epic Games, Inc. in the United States of America and elsewhere.

## Citation

```bibtex
@inproceedings{sun2026wildfirex,
  title={WildfireX-SLAM: A Large-Scale Low-Altitude RGB-D Dataset for Wildfire SLAM and Beyond},
  author={Sun, Zhicong and Lo, Jacqueline and Hu, Jinxing},
  booktitle={International Conference on Multimedia Modeling},
  pages={533--546},
  year={2026},
  organization={Springer}
}
```
