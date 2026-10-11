# Sequence format

On Hugging Face each image folder is shipped as `<folder>.zip` inside the sequence folder; unzipping it in place gives
the layout below.

```
<id>/
  rgb/000000.png          640x480 BGRA/BGR uint8
  depth/000000.png        uint16, centimetres (value / 100 = metres), 0 = invalid (sky, > 655.35 m)
  normal/ ir/ albedo/ roughness/ specular/ ...   uint8 AirSim image types
  groundtruth.txt         TUM: timestamp tx ty tz qx qy qz qw (camera-to-world)
  traj.txt                Replica: one row-major 4x4 camera-to-world matrix per line
  rgb.txt depth.txt associations.txt   TUM / ORB-SLAM3 association files
  motion.csv              timestamp, position, velocity, acceleration (finite differences)
  intrinsics.yaml         width height fx fy cx cy depth_scale fps
  meta.json               id, alias, benchmark, trajectory id, full resolved config, versions, scene objects
  frame_qa.json qa.json   per-frame / per-sequence quality checks
```

- **Pose convention**: OpenCV camera (x right, y down, z forward); world = AirSim NED (x north, y east, z down), metres.
- **Intrinsics**: pinhole, FOV 90° → fx = fy = 320, cx = 320, cy = 240.
- **Timestamps**: nominal k/fps seconds. Between two frames the world (fire, smoke, leaves, foliage) advances exactly 1/fps s.
  Five pre-roll captures at the start pose are dropped before frame 0 (`meta.json: preroll_frames`).
- **Depth content**: z-depth of the rendered scene at the same instant as RGB. Smoke from the fire VFX and
  falling leaves are written into depth; foliage moves with wind. Source precision is float16
  (≈0.8 cm at 10 m, ≈3 cm at 50 m).
- `ir` is AirSim's object-ID based infrared, not a physical thermal image.
