"""Top-surface (canopy) survey and automatic fire-site selection (WildfireX-Sim 1.1+).

survey(): fire off, sun at noon, camera looking straight down on a grid; every depth map is back-projected into
world NED points and rasterised into a height grid keeping the highest surface per cell. Output: surface.npz
(x0, y0, step, z) in NED metres, z down. Smoke is excluded because all fire actors are parked during the survey.

pick_sites(): forested (canopy 6-45 m above ground), moderate slope, well separated, inside the map.
"""
import math
import time

import numpy as np

from ..geometry import camera_c2w_cv, intrinsics
from ..sim.client import airsim

VEHICLE = "UAV1"
CAMERA = "rgbd"


def survey(client, scene, xlim, ylim, out_path, altitude=180.0, spacing=200.0, cell=2.0, settle=2.0,
           width=640, height=480, fov=90.0, log=print):
    scene.set_fire({"mode": "off"})
    scene.set_weather({})
    scene.set_light({"preset": "noon"})
    K = intrinsics(width, height, fov)
    xs = np.arange(xlim[0], xlim[1] + 1e-6, spacing)
    ys = np.arange(ylim[0], ylim[1] + 1e-6, spacing)
    nx = int(math.ceil((xlim[1] - xlim[0]) / cell)) + 1
    ny = int(math.ceil((ylim[1] - ylim[0]) / cell)) + 1
    grid = np.full((nx, ny), np.inf)
    v, u = np.mgrid[0:height:2, 0:width:2]
    rays = np.stack([(u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"], np.ones_like(u, float)], -1).reshape(-1, 3)
    down = airsim.to_quaternion(math.radians(-90), 0, 0)
    t0 = time.time()
    for i, x in enumerate(xs):
        for j, y in enumerate(ys if i % 2 == 0 else ys[::-1]):
            gz = float(scene.world.ground_z(x, y))
            client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(float(x), float(y), gz - altitude), down), True, VEHICLE)
            scene.run_world(settle)  # let world-partition streaming load the new area
            r = client.simGetImages([airsim.ImageRequest(CAMERA, 1, pixels_as_float=True, compress=False)], VEHICLE)[0]
            d = np.array(r.image_data_float, np.float32).reshape(r.height, r.width)[0:height:2, 0:width:2].reshape(-1)
            c2w = camera_c2w_cv(client.simGetCameraInfo(CAMERA, VEHICLE).pose)
            ok = (d > 0) & (d < altitude * 3)
            P = (rays[ok] * d[ok, None]) @ c2w[:3, :3].T + c2w[:3, 3]
            ii = np.round((P[:, 0] - xlim[0]) / cell).astype(int)
            jj = np.round((P[:, 1] - ylim[0]) / cell).astype(int)
            m = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
            np.minimum.at(grid, (ii[m], jj[m]), P[m, 2])
        log(f"  survey row {i + 1}/{len(xs)}  {time.time() - t0:.0f}s")
    grid[~np.isfinite(grid)] = np.nan
    # store in UE world coordinates (cm): valid across builds/sessions whose AirSim NED origins differ
    x0u, y0u, _ = scene.world.ned_to_ue([xlim[0], ylim[0], 0.0])
    z_cm = scene.world.origin[2] - grid * 100.0
    np.savez_compressed(out_path, frame="ue", x0_cm=x0u, y0_cm=y0u, step_cm=cell * 100.0, z_cm=z_cm.astype(np.float32))
    scene.restore()
    return out_path


def pick_sites(world, n=5, min_sep=350.0, margin=300.0, exclude=(), seed=0, radius=40.0):
    """Choose fire sites in forest: mean canopy height 6-45 m within `radius`, slope < 20 deg, >= min_sep apart."""
    rng = np.random.default_rng(seed)
    (x0, x1), (y0, y1) = world.bounds_ned()
    cand = []
    for _ in range(4000):
        x, y = rng.uniform(x0 + margin, x1 - margin), rng.uniform(y0 + margin, y1 - margin)
        ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        px, py = x + radius * np.cos(ang), y + radius * np.sin(ang)
        g = world.ground_z(px, py)
        s = world.surface_z(px, py)
        canopy = float(np.nanmean(g - s))  # NED: ground z - surface z = height above ground
        g0 = float(world.ground_z(x, y))
        slope = math.degrees(math.atan(float(np.nanmax(np.abs(g - g0))) / radius))
        if not (6.0 <= canopy <= 45.0) or slope > 20.0 or not np.isfinite(canopy):
            continue
        cand.append((x, y, canopy, slope))
    chosen = [np.array(e, float) for e in exclude]
    out = []
    for x, y, canopy, slope in sorted(cand, key=lambda c: c[3]):  # flattest first
        if all(np.hypot(x - c[0], y - c[1]) >= min_sep for c in chosen):
            chosen.append(np.array([x, y]))
            u = world.ned_to_ue([x, y, float(world.ground_z(x, y))])
            out.append({"xy_ue_cm": [round(float(u[0]), 1), round(float(u[1]), 1)], "ground_z_ue_cm": round(float(u[2]), 1),
                        "canopy_height_m": round(canopy, 1), "slope_deg": round(slope, 1)})
        if len(out) >= n:
            break
    return out


def refine(client, scene, surface_path, centers, half=90.0, above=40.0, spacing=25.0, settle=2.5,
           width=640, height=480, fov=90.0, log=print):
    """Re-survey the canopy around fire sites from low altitude and raise the surface map where it was too low.

    From 180 m the renderer drops thin foliage and uses coarse LODs, so tall crowns can be underestimated by
    10 m or more; flights that follow that surface then clip tree tops. Here the camera looks down from `above`
    metres over the current surface on a `spacing` grid covering +-`half` m around each centre (NED xy), with the
    world running so full-detail foliage streams in. Cells keep the highest surface seen by either survey.
    """
    scene.set_fire({"mode": "off"})
    scene.set_weather({})
    scene.set_light({"preset": "noon"})
    w = scene.world
    w.load_surface(surface_path)
    d0 = dict(np.load(surface_path))
    z_cm = d0["z_cm"].astype(np.float64)
    x0u, y0u, step = float(d0["x0_cm"]), float(d0["y0_cm"]), float(d0["step_cm"])
    nx, ny = z_cm.shape
    K = intrinsics(width, height, fov)
    v, u = np.mgrid[0:height:2, 0:width:2]
    rays = np.stack([(u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"], np.ones_like(u, float)], -1).reshape(-1, 3)
    down = airsim.to_quaternion(math.radians(-90), 0, 0)
    raised = np.zeros_like(z_cm, bool)
    t0 = time.time()
    for ci, (cx, cy) in enumerate(centers):
        offs = np.arange(-half, half + 1e-6, spacing)
        for i, dx in enumerate(offs):
            for dy in (offs if i % 2 == 0 else offs[::-1]):
                x, y = cx + dx, cy + dy
                z = float(w.surface_z(x, y)) - above
                client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(float(x), float(y), z), down), True, VEHICLE)
                scene.run_world(settle)
                r = client.simGetImages([airsim.ImageRequest(CAMERA, 1, pixels_as_float=True, compress=False)], VEHICLE)[0]
                d = np.array(r.image_data_float, np.float32).reshape(r.height, r.width)[0:height:2, 0:width:2].reshape(-1)
                c2w = camera_c2w_cv(client.simGetCameraInfo(CAMERA, VEHICLE).pose)
                ok = (d > 0.5) & (d < above * 4)
                P = (rays[ok] * d[ok, None]) @ c2w[:3, :3].T + c2w[:3, 3]
                U = w.ned_to_ue(P)
                ii = np.round((U[:, 0] - x0u) / step).astype(int)
                jj = np.round((U[:, 1] - y0u) / step).astype(int)
                m = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
                ii, jj, zu = ii[m], jj[m], U[m, 2]
                before = z_cm[ii, jj].copy()
                np.fmax.at(z_cm, (ii, jj), zu)  # UE z up: keep the highest surface
                raised[ii, jj] |= z_cm[ii, jj] > np.nan_to_num(before, nan=-1e12) + 1.0
        log(f"  refine site {ci + 1}/{len(centers)}  {time.time() - t0:.0f}s")
    np.savez_compressed(surface_path, frame="ue", x0_cm=x0u, y0_cm=y0u, step_cm=step, z_cm=z_cm.astype(np.float32))
    w.load_surface(surface_path)
    scene.restore()
    return int(raised.sum())
