"""Waypoint generators (zigzag / spiral / search, ported from the WildfireX-SLAM v1 notebook)
and densification into per-frame poses at constant speed (positions linear, orientations slerp).

Coordinates are AirSim NED (metres); quaternions are (w, x, y, z).
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def euler_to_quat_wxyz(pitch, roll, yaw):
    # same convention as airsim.to_quaternion(pitch, roll, yaw): body ZYX (yaw -> pitch -> roll)
    q = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()  # x y z w
    return np.array([q[3], q[0], q[1], q[2]])


def look_at_quat(p_from, p_to):
    d = np.asarray(p_to, float) - np.asarray(p_from, float)
    yaw = math.atan2(d[1], d[0])
    pitch = -math.atan2(d[2], math.hypot(d[0], d[1]))  # NED: target below -> look down (negative pitch)
    return euler_to_quat_wxyz(pitch, 0.0, yaw)


# ---------------- waypoints ----------------
def zigzag(center, height, width, length, spacing, pitch_deg=-90.0, yaw_deg=0.0):
    cx, cy, cz = center
    nx = int(width / spacing) + 1
    ny = int(length / spacing) + 1
    q = euler_to_quat_wxyz(math.radians(pitch_deg), 0.0, math.radians(yaw_deg))
    pts, quats = [], []
    for i in range(ny):
        y = cy - length / 2 + i * spacing
        xs = np.linspace(cx - width / 2, cx + width / 2, nx)
        if i % 2:
            xs = xs[::-1]
        for x in xs:
            pts.append((x, y, cz - height))
            quats.append(q)
    return np.array(pts), np.array(quats)


def spiral(center, radius, turns, phi_start_deg, phi_end_deg, n_points=360):
    """Spherical spiral around `center`; the camera always looks at the center."""
    cx, cy, cz = center
    pts, quats = [], []
    for k in range(n_points):
        t = k / (n_points - 1)
        theta = 2 * math.pi * turns * t
        phi = math.radians(phi_start_deg + (phi_end_deg - phi_start_deg) * t)
        p = (cx + radius * math.sin(phi) * math.cos(theta),
             cy + radius * math.sin(phi) * math.sin(theta),
             cz - radius * math.cos(phi))
        pts.append(p)
        quats.append(look_at_quat(p, center))
    return np.array(pts), np.array(quats)


def search_path(center, start_offset, end_height, curvature=0.3, n_ctrl=None, seed=0, look_at_center=True):
    """Random smooth Bezier path from a start offset to above the center (search-scan)."""
    rng = np.random.default_rng(seed)
    cx, cy, cz = center
    p0 = np.array([cx + start_offset[0], cy + start_offset[1], cz - start_offset[2]])
    p3 = np.array([cx, cy, cz - end_height])
    L = np.linalg.norm(p3 - p0)
    n_ctrl = n_ctrl or int(rng.integers(3, 6))
    ctrl = [p0]
    for i in range(1, n_ctrl - 1):
        t = i / (n_ctrl - 1)
        base = p0 + t * (p3 - p0)
        off = np.r_[rng.uniform(-1, 1, 2) * L * 0.4, rng.uniform(-1, 1) * L * 0.3] * curvature * math.sin(t * math.pi)
        ctrl.append(base + off)
    ctrl.append(p3)
    ctrl = np.array(ctrl)
    ts = np.linspace(0, 1, 400)
    pts = np.array([_bezier(ctrl, t) for t in ts])
    if look_at_center:
        quats = np.array([look_at_quat(p, center) for p in pts])
    else:
        quats = _heading_quats(pts, pitch=math.radians(-30))
    return pts, quats


def _bezier(ctrl, t):
    p = ctrl.astype(float).copy()
    for r in range(1, len(p)):
        p[:len(p) - r] = (1 - t) * p[:len(p) - r] + t * p[1:len(p) - r + 1]
    return p[0]


def _heading_quats(pts, pitch):
    quats = []
    for k in range(len(pts)):
        d = pts[min(k + 1, len(pts) - 1)] - pts[max(k - 1, 0)]
        quats.append(euler_to_quat_wxyz(pitch, 0.0, math.atan2(d[1], d[0])))
    return np.array(quats)


# ---------------- densification ----------------
def densify(points, quats, speed=3.0, fps=30, max_yaw_rate_deg=45.0):
    """
    Resample by arc length at constant speed: speed/fps metres per frame (0.1 m by default).
    Rotations between waypoints are limited to max_yaw_rate so the camera never snaps.
    Returns (N,3) positions and (N,4) wxyz quaternions.
    """
    points = np.asarray(points, float)
    quats = np.asarray(quats, float)
    step = speed / fps
    max_rot = math.radians(max_yaw_rate_deg) / fps
    out_p, out_q = [points[0]], [quats[0]]
    for a in range(len(points) - 1):
        p0, p1 = points[a], points[a + 1]
        r0 = Rotation.from_quat(np.roll(quats[a], -1))
        r1 = Rotation.from_quat(np.roll(quats[a + 1], -1))
        ang = (r0.inv() * r1).magnitude()
        dist = np.linalg.norm(p1 - p0)
        n = max(1, int(math.ceil(dist / step)), int(math.ceil(ang / max_rot)))
        slerp = Slerp([0, 1], Rotation.concatenate([r0, r1]))
        for k in range(1, n + 1):
            t = k / n
            out_p.append(p0 + t * (p1 - p0))
            out_q.append(np.roll(slerp([t]).as_quat()[0], 1))
    return np.array(out_p), np.array(out_q)


def path_length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def follow_surface(points, world, ref, clearance, fps=30, radius=6.0, smooth_s=1.5):
    """Re-set altitudes so the camera stays `clearance` metres above the canopy (ref='canopy', from a survey)
    or the terrain (ref='ground'), checking a disc of `radius` m around each point. Altitude changes are
    smoothed over ~smooth_s seconds; the result never goes below the required clearance. NED: z down."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    pts = np.asarray(points, float).copy()
    f = world.surface_z if ref == "canopy" else world.ground_z
    r, d = radius, radius * 0.7071
    offs = [(0, 0), (r, 0), (-r, 0), (0, r), (0, -r), (d, d), (-d, d), (d, -d), (-d, -d)]
    top = np.min([f(pts[:, 0] + dx, pts[:, 1] + dy) for dx, dy in offs], axis=0)  # smaller NED z = higher
    need = top - clearance
    w = max(1, int(smooth_s * fps))
    safe = minimum_filter1d(need, size=4 * w + 1, mode="nearest")
    smooth = uniform_filter1d(safe, size=2 * w + 1, mode="nearest")
    pts[:, 2] = np.minimum(smooth, need)
    return pts
