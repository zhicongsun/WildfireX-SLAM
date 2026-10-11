"""Writers for the per-sequence files (pose files, association lists, intrinsics, motion)."""
import os

import numpy as np

from ..geometry import R_to_quat_xyzw

DEPTH_SCALE = 100.0  # depth png = metres * 100 (cm); max 655.35 m; 0 = invalid (sky / out of range)


def depth_to_u16(d):
    d = np.where(np.isfinite(d) & (d > 0) & (d < 65535 / DEPTH_SCALE), d, 0)
    return np.round(d * DEPTH_SCALE).astype(np.uint16)


def write_replica_traj(path, c2ws):
    with open(path, "w") as f:
        for T4 in c2ws:
            f.write(" ".join(f"{v:.8f}" for v in T4.reshape(-1)) + "\n")


def write_tum(path, timestamps, c2ws):
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw  (camera-to-world, OpenCV camera, world = AirSim NED)\n")
        for t, T4 in zip(timestamps, c2ws):
            qx, qy, qz, qw = R_to_quat_xyzw(T4[:3, :3])
            f.write(f"{t:.6f} {T4[0, 3]:.6f} {T4[1, 3]:.6f} {T4[2, 3]:.6f} {qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}\n")


def write_associations(out, timestamps):
    for m in ("rgb", "depth"):
        with open(os.path.join(out, f"{m}.txt"), "w") as f:
            for k, t in enumerate(timestamps):
                f.write(f"{t:.6f} {m}/{k:06d}.png\n")
    with open(os.path.join(out, "associations.txt"), "w") as f:
        for k, t in enumerate(timestamps):
            f.write(f"{t:.6f} rgb/{k:06d}.png {t:.6f} depth/{k:06d}.png\n")


def write_motion(path, timestamps, c2ws, fps):
    P = c2ws[:, :3, 3]
    n = len(P)
    V = np.gradient(P, 1.0 / fps, axis=0) if n > 1 else np.zeros_like(P)
    A = np.gradient(V, 1.0 / fps, axis=0) if n > 2 else np.zeros_like(P)
    with open(path, "w") as f:
        f.write("timestamp,x,y,z,vx,vy,vz,ax,ay,az\n")
        for k in range(n):
            f.write(",".join(f"{v:.6f}" for v in [timestamps[k], *P[k], *V[k], *A[k]]) + "\n")


def write_intrinsics(path, K, fps, depth_scale=DEPTH_SCALE):
    with open(path, "w") as f:
        f.write(f"width: {K['width']}\nheight: {K['height']}\nfx: {K['fx']:.6f}\nfy: {K['fy']:.6f}\n"
                f"cx: {K['cx']:.6f}\ncy: {K['cy']:.6f}\ndepth_scale: {depth_scale}\nfps: {fps}\n")


def write_all(out, c2ws, fps, K):
    """Pose files, association lists, motion and intrinsics for one sequence."""
    ts = np.arange(len(c2ws)) / fps
    write_replica_traj(os.path.join(out, "traj.txt"), c2ws)
    write_tum(os.path.join(out, "groundtruth.txt"), ts, c2ws)
    write_associations(out, ts)
    write_motion(os.path.join(out, "motion.csv"), ts, c2ws, fps)
    write_intrinsics(os.path.join(out, "intrinsics.yaml"), K, fps)
