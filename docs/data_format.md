# Sequence format

On Hugging Face each image folder is shipped as `<folder>.zip` inside the sequence folder; unzipping it in place gives
the layout below.

```
<id>/
  rgb/000000.png          640x480 8-bit RGB PNG
  depth/000000.png        uint16, centimetres (value / 100 = metres), 0 = invalid (sky, > 655.35 m)
  normal/ albedo/         8-bit RGB PNG
  roughness/ specular/ ir/   8-bit grey values stored as 3 identical channels
  groundtruth.txt         TUM: timestamp tx ty tz qx qy qz qw (camera-to-world)
  traj.txt                Replica: one row-major 4x4 camera-to-world matrix per line
  rgb.txt depth.txt associations.txt   TUM / ORB-SLAM3 association files
  motion.csv              timestamp, position, velocity, acceleration (finite differences)
  intrinsics.yaml         width height fx fy cx cy depth_scale fps
  meta.json               id, alias, benchmark, trajectory id, full resolved config, versions, scene objects
  frame_qa.json qa.json   per-frame / per-sequence quality checks
```

## Conventions

| | |
|---|---|
| **Depth** | 16-bit PNG, planar depth in **centimetres**: metres = value / 100 (`depth_scale: 100` in `intrinsics.yaml`); 0 = invalid. **This is not TUM's 5000** — set the depth scale of your loader to **100**, otherwise all depths come out 50× too small. Centimetres are needed because depth reaches several hundred metres outdoors (TUM's 5000 would cap it at 13 m). |
| **Camera frame** | OpenCV: x right, y down, z forward — the same as TUM and Replica. |
| **Poses** | camera-to-world. `groundtruth.txt`: TUM format `timestamp tx ty tz qx qy qz qw`; `traj.txt`: Replica format, one row-major 4×4 matrix per line. |
| **World frame** | AirSim NED: x north, y east, **z down** (metres, right-handed). All 19 sequences share one world frame. Trajectory metrics (ATE/RPE with alignment) are unaffected; viewers that assume z up show the scene upside down. |
| **Intrinsics** | pinhole, no distortion: fx = fy = 320, cx = 320, cy = 240 (640 × 480). |

To get a z-up world (x north, y west, z up), left-multiply every pose; the camera frame stays OpenCV:

```python
import numpy as np
F = np.diag([1.0, -1.0, -1.0, 1.0])   # NED -> NWU (z up), still right-handed
c2w_zup = F @ c2w
```

## Notes

- **Timestamps**: nominal k/fps seconds. Between two frames the world (fire, smoke, leaves, foliage) advances exactly 1/fps s.
  Five pre-roll captures at the start pose are dropped before frame 0 (`meta.json: preroll_frames`).
- **Depth content**: z-depth of the rendered scene at the same instant as RGB. Smoke from the fire VFX and
  falling leaves are written into depth; foliage moves with wind. Source precision is float16
  (≈0.8 cm at 10 m, ≈3 cm at 50 m).
- `ir` is AirSim's object-ID based infrared, not a physical thermal image.
