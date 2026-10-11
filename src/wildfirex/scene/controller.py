"""Runtime scene control. Nothing here is saved to the level: every change lives only in the running game.

Works with WildfireX-Sim 1.0 (fire on/off, sun, weather) and 1.1 (adds movable fire kits, a moon and night
exposure; actor names come from sim/resources/wildfirex_sim_1.1.json).
"""
import json
import math
import os
import time

import numpy as np

from ..sim.client import airsim
from .presets import (FIRE_HIDE_DROP, GLOW_DEPTH_M, GLOW_FULL_DEPTH_M, MOON_PITCH_DEG, MOON_PARK_PITCH_DEG, MOON_YAW_DEG,
                      SUN_PRESETS, SUN_YAW_DEG, SUN_YAW_PRESETS)
from .worldmap import RESOURCES, WorldMap, load_manifest

IDENTITY = airsim.Quaternionr(0.0, 0.0, 0.0, 1.0)


def _pose(xyz, q=IDENTITY):
    return airsim.Pose(airsim.Vector3r(*map(float, xyz)), q)


def load_sites(name="sites_sim_1.1.json"):
    path = os.path.join(RESOURCES, name)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Scene:
    def __init__(self, client, game_speed):
        self.c = client
        self.game_speed = game_speed
        names = set(client.simListSceneObjects())
        manifest = load_manifest()
        same_level = bool(manifest) and manifest["reference"]["actor"] in names
        # kits / moon exist only in 1.1; the ground map and sites are valid for any build of the same level
        self.m = manifest if same_level and manifest["moon"] in names else None
        self.world = WorldMap.from_client(client, manifest) if same_level else None
        self.sites = load_sites() if same_level else {}
        if same_level:
            self.fires = [n for n in manifest["original_fire_actors"] if n in names]
        else:
            self.fires = sorted(n for n in names if "niagara" in n.lower())
        self.version = self.m["simulator"] if self.m else "WildfireX-Sim 1.0"
        self.fire_pose0 = {f: client.simGetObjectPose(f) for f in self.fires}
        self.fire_pos = None
        if self.fires:
            p = self.fire_pose0[self.fires[0]].position
            self.fire_pos = np.array([p.x_val, p.y_val, p.z_val])
        self.center = None if self.fire_pos is None else self.fire_pos.copy()  # trajectory center (set by set_fire)
        moon = self.m["moon"] if self.m else None
        self.moon = moon if moon in names else None
        self.sun = next((n for n in sorted(names) if n.lower().startswith("directionallight") and n != moon), None)
        self.sun_pose0 = client.simGetObjectPose(self.sun) if self.sun else None
        self.moon_pose0 = client.simGetObjectPose(self.moon) if self.moon else None
        self.kits_placed = []
        self._light_preset = None

    # ---- sites ----
    def site_xy(self, site):
        if site is None or site == "ed01":
            return self.fire_pos[:2].copy()
        if isinstance(site, (list, tuple)):
            return np.array(site[:2], float)
        if site not in self.sites:
            raise KeyError(f"unknown fire site {site!r}; known: {['ed01', *sorted(self.sites)]}")
        v = self.sites[site]
        if isinstance(v, dict):
            return self.world.site_xy_ned(v)
        return np.array(v[:2], float)  # config-provided [x, y] in this session's NED

    # ---- fire ----
    def _park_kits(self):
        if not self.m:
            return
        park = self.world.ue_to_ned(self.m["park_ue_cm"])
        for kit in self.m["kits"].values():
            for name in [*kit["flames"].values(), kit["smoke_seen"], kit["smoke_unseen"], kit["light"]]:
                self.c.simSetObjectPose(name, _pose(park), True)
        self.kits_placed = []

    def _set_original(self, on):
        for f, p0 in self.fire_pose0.items():
            z = p0.position.z_val + (0.0 if on else FIRE_HIDE_DROP)
            self.c.simSetObjectPose(f, airsim.Pose(airsim.Vector3r(p0.position.x_val, p0.position.y_val, z), p0.orientation), True)

    def set_fire(self, fire):
        mode = fire.get("mode", "static")
        if mode == "spread":
            raise NotImplementedError("fire.mode 'spread' is planned for WildfireX-Sim 2.x (see docs/architecture.md) and is not available yet")
        site = fire.get("site")
        kits = fire.get("kits") or []
        if kits and not self.m:
            raise RuntimeError("fire.kits needs WildfireX-Sim 1.1 or later")
        if site not in (None, "ed01") and not self.m:
            raise RuntimeError("fire.site other than ed01 needs WildfireX-Sim 1.1 or later")
        self._park_kits()
        xy = self.site_xy(site) if self.fire_pos is not None else None
        gz = float(self.world.ground_z(xy[0], xy[1])) if self.world is not None and xy is not None else (
            None if self.fire_pos is None else float(self.fire_pos[2]))
        self.center = None if xy is None else np.array([xy[0], xy[1], gz])
        if mode == "off":
            self._set_original(False)
            return
        keep_original = fire.get("original_smoke", site in (None, "ed01"))
        self._set_original(bool(keep_original))
        if not kits:
            return
        heading = math.radians(fire.get("heading_deg", 0.0))
        spacing = float(fire.get("spacing", 12.0))
        d = np.array([math.cos(heading), math.sin(heading)])
        variant = fire.get("flame", "NFire_Grd_00")
        for idx, kname in enumerate(kits):
            kit = self.m["kits"][kname]
            off = (idx - (len(kits) - 1) / 2.0) * spacing * d
            x, y = xy + off
            z = float(self.world.ground_z(x, y)) + float(fire.get("z_offset", 0.0))
            flame = kit["flames"].get(variant) or next(iter(kit["flames"].values()))
            for name in (flame, kit["smoke_seen"], kit["smoke_unseen"]):
                self.c.simSetObjectPose(name, _pose([x, y, z]), True)
            depth = self._glow_depth(fire.get("glow", "auto"))
            if depth is not None:
                self.c.simSetObjectPose(kit["light"], _pose([x, y, z + depth]), True)  # NED: +z = down
            self.kits_placed.append({"kit": kname, "flame": flame, "glow": depth is not None, "glow_depth_m": depth,
                                     "xyz_ned": [round(x, 3), round(y, 3), round(z, 3)]})

    def _glow_depth(self, glow):
        """fire.glow -> light depth below ground (m) or None: auto (by light preset) / true / false / number (depth)."""
        if glow is True:
            return GLOW_FULL_DEPTH_M
        if glow is False or glow is None:
            return None
        if isinstance(glow, (int, float)):
            return float(glow)
        return GLOW_DEPTH_M.get(self._light_preset)

    # ---- weather ----
    def set_weather(self, weather):
        fog, leaf, dust = (float(weather.get(k, 0.0)) for k in ("fog", "leaf", "dust"))
        self.c.simEnableWeather(bool(fog or leaf or dust))
        W = airsim.WeatherParameter
        self.c.simSetWeatherParameter(W.Fog, fog)
        self.c.simSetWeatherParameter(W.MapleLeaf, leaf)
        self.c.simSetWeatherParameter(W.Dust, dust)

    # ---- light ----
    def _set_moon(self, on):
        if not self.moon:
            return
        pitch = MOON_PITCH_DEG if on else MOON_PARK_PITCH_DEG
        q = airsim.to_quaternion(math.radians(pitch), 0, math.radians(MOON_YAW_DEG))
        self.c.simSetObjectPose(self.moon, airsim.Pose(self.moon_pose0.position, q), True)

    def set_light(self, light):
        """light: {} / {"preset": name} / {"sun_pitch_deg": value}; optional "moon": bool (default: on for night)"""
        if self.sun is None:
            return
        pitch = light.get("sun_pitch_deg")
        if pitch is None and light.get("preset"):
            pitch = SUN_PRESETS[light["preset"]]
        self._set_moon(light.get("moon", light.get("preset") == "night"))
        if pitch is None:
            self.c.simSetObjectPose(self.sun, self.sun_pose0, True)
            return
        yaw = light.get("sun_yaw_deg", SUN_YAW_PRESETS.get(light.get("preset"), SUN_YAW_DEG))
        q = airsim.to_quaternion(math.radians(pitch), 0, math.radians(yaw))
        self.c.simSetObjectPose(self.sun, airsim.Pose(self.sun_pose0.position, q), True)

    def apply(self, seq):
        self._light_preset = seq["light"].get("preset")
        self.set_fire(seq["fire"])
        self.set_weather(seq["weather"])
        self.set_light(seq["light"])

    def run_world(self, seconds):
        """Run the world at normal speed (warm-up: smoke column grows, leaves start falling)."""
        self.c.simSetGameSpeed(1.0)
        self.c.simPause(False)
        time.sleep(seconds)
        self.c.simPause(True)
        self.c.simSetGameSpeed(self.game_speed)

    def restore(self):
        self._park_kits()
        self._set_original(True)
        self.set_weather({})
        self.set_light({})
        self.center = None if self.fire_pos is None else self.fire_pos.copy()
