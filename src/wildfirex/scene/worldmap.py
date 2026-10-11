"""World geometry for a simulator build: UE <-> AirSim NED transform, ground height map, canopy (surface) map.

AirSim NED = (UE - origin) * [1, 1, -1] / 100  (axes aligned, metres). The origin is recovered at runtime from a
reference actor whose UE location is recorded in the simulator manifest.
"""
import json
import os

import numpy as np

RESOURCES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim", "resources")


def load_manifest(name="wildfirex_sim_1.1.json"):
    path = os.path.join(RESOURCES, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _bilinear(grid, fi, fj):
    n0, n1 = grid.shape
    i0 = np.clip(np.floor(fi).astype(int), 0, n0 - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, n1 - 2)
    di, dj = np.clip(fi - i0, 0, 1), np.clip(fj - j0, 0, 1)
    g = grid
    return ((1 - di) * (1 - dj) * g[i0, j0] + di * (1 - dj) * g[i0 + 1, j0]
            + (1 - di) * dj * g[i0, j0 + 1] + di * dj * g[i0 + 1, j0 + 1])


class WorldMap:
    def __init__(self, manifest, origin_ue_cm):
        self.m = manifest
        self.origin = np.asarray(origin_ue_cm, float)
        gr = manifest["ground"]
        self.g_x0, self.g_y0, self.g_step = gr["x0_cm"], gr["y0_cm"], gr["step_cm"]
        self.ground_ue = np.array([[np.nan if z is None else z for z in row] for row in gr["z_cm"]], float)
        self.surface = None  # canopy / top surface, NED metres (see load_surface)

    @classmethod
    def from_client(cls, client, manifest):
        ref = manifest["reference"]
        p = client.simGetObjectPose(ref["actor"]).position
        ned = np.array([p.x_val, p.y_val, p.z_val])
        if not np.all(np.isfinite(ned)):
            raise RuntimeError(f"reference actor {ref['actor']} not found in the running simulator")
        ue = np.asarray(ref["ue_cm"], float)
        origin = ue - ned * np.array([100.0, 100.0, -100.0])
        return cls(manifest, origin)

    # ---- transforms ----
    def ue_to_ned(self, ue):
        ue = np.asarray(ue, float)
        return (ue - self.origin) * np.array([0.01, 0.01, -0.01])

    def ned_to_ue(self, ned):
        ned = np.asarray(ned, float)
        return ned * np.array([100.0, 100.0, -100.0]) + self.origin

    # ---- heights (NED z, metres, down positive) ----
    def ground_z(self, x, y):
        ue = self.ned_to_ue(np.stack([np.asarray(x, float), np.asarray(y, float), np.zeros_like(np.asarray(x, float))], -1))
        fi = (ue[..., 0] - self.g_x0) / self.g_step
        fj = (ue[..., 1] - self.g_y0) / self.g_step
        z_ue = _bilinear(self.ground_ue, fi, fj)
        return -(z_ue - self.origin[2]) / 100.0

    def load_surface(self, path):
        """Surface map from `wildfirex survey`, stored in UE world coordinates (cm) so it is valid for every
        build and session of the same level (the AirSim NED origin differs between builds)."""
        d = np.load(path)
        if "z_cm" not in d:
            raise ValueError(f"{path}: expected a UE-frame surface map (keys x0_cm, y0_cm, step_cm, z_cm)")
        self.surface = {"x0": float(d["x0_cm"]), "y0": float(d["y0_cm"]), "step": float(d["step_cm"]), "z": d["z_cm"]}

    def surface_z(self, x, y):
        """Top surface (canopy or ground) NED z from a survey; falls back to the ground map."""
        g = self.ground_z(x, y)
        if self.surface is None:
            return g
        s = self.surface
        xa, ya = np.asarray(x, float), np.asarray(y, float)
        ue = self.ned_to_ue(np.stack([xa, ya, np.zeros_like(xa)], -1))
        z_ue = _bilinear(s["z"], (ue[..., 0] - s["x0"]) / s["step"], (ue[..., 1] - s["y0"]) / s["step"])
        z = -(z_ue - self.origin[2]) / 100.0
        return np.where(np.isfinite(z), np.minimum(z, g), g)  # NED: smaller z = higher

    def site_xy_ned(self, site):
        """Site record -> NED xy. Records hold UE world coordinates (xy_ue_cm) or session NED (xy_ned)."""
        if "xy_ue_cm" in site:
            u = site["xy_ue_cm"]
            return self.ue_to_ned([u[0], u[1], 0.0])[:2]
        return np.array(site["xy_ned"], float)

    def bounds_ned(self):
        n = self.ground_ue.shape[0]
        a = self.ue_to_ned([self.g_x0, self.g_y0, 0])
        b = self.ue_to_ned([self.g_x0 + (n - 1) * self.g_step, self.g_y0 + (n - 1) * self.g_step, 0])
        return (min(a[0], b[0]), max(a[0], b[0])), (min(a[1], b[1]), max(a[1], b[1]))
