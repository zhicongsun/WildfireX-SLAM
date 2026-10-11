"""Sequence quality checks: missing frames, valid depth, camera-inside-geometry frames, and
depth reprojection consistency between consecutive frames (which also validates the pose convention)."""
import json
import os

import cv2
import numpy as np

from ..io.sequence import load_intrinsics

THRESH = {
    "valid_depth_mean": 0.75,       # oblique views see sky / >655 m background
    "near_frame_ratio": 0.01,       # share of frames with >20% of pixels closer than 0.5 m
    "reproj_rel_err_median": 0.02,  # median relative depth error after reprojecting frame k into k+1
    "first_pair_bad_excess": 0.05,  # frame 0 -> 1: share of pixels off by > 20 %, minus the typical share of later pairs
}


def reproj_error(seq, K, c2w, a, b, stride=4, bad_ratio=False):
    s = K["depth_scale"]
    da = cv2.imread(os.path.join(seq, "depth", f"{a:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float64) / s
    db = cv2.imread(os.path.join(seq, "depth", f"{b:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float64) / s
    h, w = da.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = da[v, u]
    m = z > 0
    u, v, z = u[m], v[m], z[m]
    pc = np.stack([(u - K["cx"]) / K["fx"] * z, (v - K["cy"]) / K["fy"] * z, z, np.ones_like(z)])
    pb = np.linalg.inv(c2w[b]) @ c2w[a] @ pc
    zb = pb[2]
    ok = zb > 0.1
    ub = np.round(pb[0, ok] / zb[ok] * K["fx"] + K["cx"]).astype(int)
    vb = np.round(pb[1, ok] / zb[ok] * K["fy"] + K["cy"]).astype(int)
    zb = zb[ok]
    inb = (ub >= 0) & (ub < w) & (vb >= 0) & (vb < h)
    meas, pred = db[vb[inb], ub[inb]], zb[inb]
    good = meas > 0
    if good.sum() < 100:
        return None
    rel = np.abs(meas[good] - pred[good]) / meas[good]
    if bad_ratio:
        return float((rel > 0.2).mean())
    return float(np.median(rel))


def check(seq):
    K = load_intrinsics(seq)
    c2w = np.loadtxt(os.path.join(seq, "traj.txt")).reshape(-1, 4, 4)
    n = len(c2w)
    with open(os.path.join(seq, "frame_qa.json")) as f:
        fq = json.load(f)
    missing = [k for k in range(n) if not (os.path.exists(os.path.join(seq, "rgb", f"{k:06d}.png"))
                                           and os.path.exists(os.path.join(seq, "depth", f"{k:06d}.png")))]
    steps = np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1)
    pairs = list(range(0, n - 1, max(1, n // 30)))
    errs = [e for e in (reproj_error(seq, K, c2w, a, a + 1) for a in pairs) if e is not None]
    # stale first frame (render target not yet updated after the warm-up) shows up as many inconsistent pixels
    bad = [e for e in (reproj_error(seq, K, c2w, a, a + 1, bad_ratio=True) for a in pairs[1:10]) if e is not None]
    first_bad = reproj_error(seq, K, c2w, 0, 1, bad_ratio=True) if n > 1 else None
    res = {
        "seq": os.path.basename(os.path.normpath(seq)),
        "n_frames": n,
        "missing_frames": len(missing),
        "step_m_median": float(np.median(steps)) if n > 1 else 0.0,
        "step_m_max": float(steps.max()) if n > 1 else 0.0,
        "valid_depth_mean": float(np.mean([q["valid_depth"] for q in fq])),
        "valid_depth_min": float(np.min([q["valid_depth"] for q in fq])),
        "near_frame_ratio": float(np.mean([q["near_ratio"] > 0.20 for q in fq])),
        "reproj_rel_err_median": float(np.median(errs)) if errs else None,
        "first_pair_bad_ratio": first_bad,
        "later_pairs_bad_ratio_median": float(np.median(bad)) if bad else None,
    }
    fails = []
    if res["missing_frames"]:
        fails.append("missing_frames")
    if res["valid_depth_mean"] < THRESH["valid_depth_mean"]:
        fails.append("valid_depth_mean")
    if res["near_frame_ratio"] > THRESH["near_frame_ratio"]:
        fails.append("near_frame_ratio")
    if res["reproj_rel_err_median"] is None or res["reproj_rel_err_median"] > THRESH["reproj_rel_err_median"]:
        fails.append("reproj_rel_err_median")
    if first_bad is not None and bad and first_bad - float(np.median(bad)) > THRESH["first_pair_bad_excess"]:
        fails.append("first_frame_inconsistent")
    res["pass"] = not fails
    res["fails"] = fails
    with open(os.path.join(seq, "qa.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res
