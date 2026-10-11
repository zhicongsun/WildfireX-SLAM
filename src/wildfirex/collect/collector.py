"""Per-sequence collection.

Every frame: (paused) set the camera pose -> run the world for 1/fps s (slowed by game_speed) -> pause ->
one simGetImages call for all modalities (same instant) -> camera pose -> asynchronous write.
"""
import json
import os
import platform
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .. import CONFIG_SCHEMA_VERSION, __version__
from ..config import MODALITY_TYPES
from ..geometry import camera_c2w_cv, intrinsics
from ..io import formats
from ..scene.controller import Scene
from ..scene.presets import GLOW_DEPTH_M, SUN_PRESETS, SUN_YAW_DEG, SUN_YAW_PRESETS
from ..sim.client import airsim
from . import trajectories as T

VEHICLE = "UAV1"
CAMERA = "rgbd"
NEAR_THRESH = 0.5  # m
PREROLL_FRAMES = 5  # captured and dropped before frame 0


def build_path(seq, fire_pos, world=None):
    """fire_pos: trajectory center (NED). world: WorldMap, needed for height_ref canopy / ground."""
    p = seq["trajectory"]
    center = np.array(fire_pos if p.get("center", "fire") == "fire" else p["center"], float)
    site = center.copy()  # before `offset`: what `look_at: fire` looks at
    center[:2] += np.array(p.get("offset", [0, 0]), float)
    kind = p["type"]
    if kind == "zigzag":
        pts, qs = T.zigzag(center, p["height"], p["width"], p["length"], p["spacing"],
                           p.get("pitch_deg", -90.0), p.get("yaw_deg", 0.0))
    elif kind == "spiral":
        pts, qs = T.spiral(center, p["radius"], p["turns"], p["phi_start_deg"], p["phi_end_deg"], p.get("n_points", 360))
    else:
        pts, qs = T.search_path(center, p["start_offset"], p["end_height"], p.get("curvature", 0.3),
                                seed=p.get("seed", 0), look_at_center=p.get("look_at_center", True))
    pts, qs = T.densify(pts, qs, speed=seq["speed"], fps=seq["fps"])
    ref = p.get("height_ref", "center")
    if ref in ("canopy", "ground"):
        if world is None:
            raise RuntimeError(f"trajectory height_ref '{ref}' needs WildfireX-Sim 1.1 world data")
        pts = T.follow_surface(pts, world, ref, clearance=float(p["clearance"]), fps=seq["fps"],
                               radius=float(p.get("safety_radius", 6.0)))
        if kind in ("spiral", "search") and p.get("look_at_center", True):
            target = (site if p.get("look_at") == "fire" else center).copy()
            target[2] = float(world.ground_z(target[0], target[1])) if world is not None else target[2]
            qs = np.array([T.look_at_quat(q, target) for q in pts])
    if seq.get("max_frames"):
        pts, qs = pts[:seq["max_frames"]], qs[:seq["max_frames"]]
    return pts, qs


def capture_size(settings_path):
    with open(settings_path, encoding="utf-8") as f:
        s = json.load(f)
    for cs in s["CameraDefaults"]["CaptureSettings"]:
        if cs["ImageType"] == 0:
            return cs["Width"], cs["Height"], cs["FOV_Degrees"]
    raise ValueError("settings: no CaptureSettings for ImageType 0")


def decode_uint8(r):
    buf = np.frombuffer(r.image_data_uint8, np.uint8)
    hw = r.height * r.width
    if hw and buf.size % hw == 0 and buf.size // hw in (1, 3, 4):
        return buf.reshape(r.height, r.width, buf.size // hw)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot decode image: {buf.size} bytes, {r.width}x{r.height}, type={r.image_type}")
    return img


class Collector:
    def __init__(self, seq, client, out_root, run_info):
        self.seq = seq
        self.c = client
        self.out = os.path.join(out_root, seq["id"])
        self.run_info = run_info
        w, h, fov = capture_size(run_info["settings"])
        self.K = intrinsics(w, h, fov)

    def run(self):
        seq, c = self.seq, self.c
        if os.path.exists(self.out):
            shutil.rmtree(self.out)
        for m in seq["modalities"]:
            os.makedirs(os.path.join(self.out, m), exist_ok=True)
        pool = ThreadPoolExecutor(6)

        scene = Scene(c, seq["game_speed"])
        if scene.fire_pos is None:
            raise RuntimeError("no Niagara fire actor found in the level")
        scene.sites.update(self.run_info.get("sites") or {})
        if scene.world is not None and self.run_info.get("surface"):
            scene.world.load_surface(self.run_info["surface"])
        c.simSetGameSpeed(seq["game_speed"])
        c.simPause(True)
        scene.apply(seq)
        self.placed = list(scene.kits_placed)
        self.center_used = None if scene.center is None else [round(float(v), 3) for v in scene.center]

        pts, qs = build_path(seq, scene.center, scene.world)
        n = len(pts)
        print(f"[{seq['id']}] {seq.get('alias', '')} {n} frames, path {T.path_length(pts):.1f} m", flush=True)
        self._set_pose(pts[0], qs[0])
        scene.run_world(seq["warmup_sec"])

        reqs = [airsim.ImageRequest(CAMERA, MODALITY_TYPES[m], pixels_as_float=(m == "depth"), compress=False)
                for m in seq["modalities"]]
        run_wall = (1.0 / seq["fps"]) / seq["game_speed"]
        # pre-roll: the first capture after the warm-up still holds stale render-target content (depth holes and
        # 10-20 % of pixels inconsistent with the next frame), so a few frames at the start pose are captured and dropped
        for _ in range(PREROLL_FRAMES):
            self._set_pose(pts[0], qs[0])
            c.simPause(False)
            time.sleep(run_wall)
            c.simPause(True)
            c.simGetImages(reqs, VEHICLE)
        c2ws, frame_qa, futures = [], [], []
        t_start = time.time()
        try:
            for k in range(n):
                self._set_pose(pts[k], qs[k])
                c.simPause(False)
                time.sleep(run_wall)
                c.simPause(True)
                rs = c.simGetImages(reqs, VEHICLE)
                c2ws.append(camera_c2w_cv(c.simGetCameraInfo(CAMERA, VEHICLE).pose))
                qa = {"k": k}
                for m, r in zip(seq["modalities"], rs):
                    path = os.path.join(self.out, m, f"{k:06d}.png")
                    if m == "depth":
                        d = np.array(r.image_data_float, np.float32).reshape(r.height, r.width)
                        u16 = formats.depth_to_u16(d)
                        valid = u16 > 0
                        qa["valid_depth"] = float(valid.mean())
                        qa["min_depth"] = float(d[valid].min()) if valid.any() else -1.0
                        qa["near_ratio"] = float(((d > 0) & (d < NEAR_THRESH)).mean())
                        futures.append(pool.submit(cv2.imwrite, path, u16))
                    else:
                        futures.append(pool.submit(cv2.imwrite, path, decode_uint8(r)))
                frame_qa.append(qa)
                if k % 100 == 0:
                    print(f"  {k}/{n}  {(time.time() - t_start) / max(k, 1):.2f} s/frame", flush=True)
            for f in futures:
                f.result()
        finally:
            pool.shutdown(wait=True)
            scene.restore()
            c.simPause(False)
            c.simSetGameSpeed(1.0)

        c2ws = np.array(c2ws)
        formats.write_all(self.out, c2ws, seq["fps"], self.K)
        self._write_meta(n, scene, (time.time() - t_start) / n)
        with open(os.path.join(self.out, "frame_qa.json"), "w") as f:
            json.dump(frame_qa, f)
        return self.out

    def _set_pose(self, p, q):
        pose = airsim.Pose(airsim.Vector3r(*map(float, p)), airsim.Quaternionr(float(q[1]), float(q[2]), float(q[3]), float(q[0])))
        self.c.simSetVehiclePose(pose, True, VEHICLE)

    def _write_meta(self, n, scene, sec_per_frame):
        seq = self.seq
        meta = {
            "id": seq["id"], "alias": seq.get("alias"), "benchmark": seq.get("benchmark"),
            "trajectory_id": seq.get("trajectory_id"), "n_frames": n, "fps": seq["fps"],
            "sequence_config": {k: v for k, v in seq.items() if not k.startswith("_")},
            "versions": {"wildfirex": __version__, "config_schema": CONFIG_SCHEMA_VERSION,
                         "simulator": self.run_info.get("simulator_version"), "release": self.run_info.get("release")},
            "simulator_executable": os.path.basename(self.run_info.get("executable", "")), "preroll_frames": PREROLL_FRAMES,
            "depth": {"type": "DepthPlanar", "unit": "cm", "scale": formats.DEPTH_SCALE, "invalid": 0,
                      "source_precision": "float16 render target (~3 cm at 50 m, ~0.8 cm at 10 m)",
                      "content": "same-frame scene depth; includes smoke written by the VFX and wind-animated foliage"},
            "pose_convention": "camera-to-world, OpenCV camera (x right, y down, z forward), world = AirSim NED (m)",
            "timestamps": f"nominal k/{seq['fps']} s; world advanced 1/{seq['fps']} s per frame via simSetGameSpeed",
            "simulator_detected": scene.version,
            "ned_origin_ue_cm": None if scene.world is None else [round(float(v), 2) for v in scene.world.origin],
            "fire_objects": scene.fires, "fire_position_ned": scene.fire_pos.tolist(),
            "trajectory_center_ned": self.center_used,
            "fire_kits_placed": self.placed,
            "sun_object": scene.sun, "moon_object": scene.moon, "sun_presets_pitch_deg": SUN_PRESETS, "sun_yaw_deg": SUN_YAW_DEG, "sun_yaw_presets_deg": SUN_YAW_PRESETS,
            "glow_depth_m": GLOW_DEPTH_M,
            "sec_per_frame": round(sec_per_frame, 3),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.platform(),
        }
        with open(os.path.join(self.out, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
