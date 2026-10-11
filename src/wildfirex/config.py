"""Collection config (YAML). See docs/config.md for the full schema.

schema_version: 1
name: <release name>
output_dir: <dir>                 # relative to the config file
simulator: {executable: ..., port: 41451}
defaults: {...}                   # merged into every sequence
trajectories: {T01: {...}, ...}   # reusable paths; paired sequences reference the same id
sequences: [{id, alias, benchmark, trajectory, fire, light, weather, ...}, ...]
"""
import copy
import os

import yaml

from . import CONFIG_SCHEMA_VERSION
from .scene.presets import FIRE_KITS, FLAME_VARIANTS, SUN_PRESETS

MODALITY_TYPES = {
    "rgb": 0, "depth": 1, "segmentation": 5, "normal": 6, "ir": 7,
    "metallic": 16, "albedo": 17, "specular": 18, "roughness": 20,
}
FIRE_MODES = ("off", "static", "spread")
TRAJECTORY_TYPES = ("zigzag", "spiral", "search")

BUILTIN_DEFAULTS = {
    "speed": 3.0,              # m/s
    "fps": 30,
    "game_speed": 0.1,         # world slow-down while a frame is advanced
    "warmup_sec": 30.0,        # world seconds at normal speed before recording
    "modalities": ["rgb", "depth"],
    "fire": {"mode": "static"},
    "light": {},
    "weather": {},
    "max_frames": None,
}


class ConfigError(ValueError):
    pass


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ver = cfg.get("schema_version")
    if ver != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"{path}: schema_version {ver!r} is not supported (expected {CONFIG_SCHEMA_VERSION})")
    base = os.path.dirname(os.path.abspath(path))
    cfg["_path"] = os.path.abspath(path)
    cfg["output_dir"] = os.path.normpath(os.path.join(base, cfg.get("output_dir", "output")))
    cfg.setdefault("simulator", {})
    cfg.setdefault("sites", {})
    cfg["sequences"] = [resolve_sequence(cfg, s) for s in cfg.get("sequences", [])]
    ids = [s["id"] for s in cfg["sequences"]]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ConfigError(f"duplicate sequence ids: {sorted(dup)}")
    return cfg


def resolve_sequence(cfg, seq):
    s = _merge(_merge(BUILTIN_DEFAULTS, cfg.get("defaults")), seq)
    if "id" not in s:
        raise ConfigError(f"sequence without id: {seq}")
    traj = s.get("trajectory")
    if isinstance(traj, str):
        table = cfg.get("trajectories", {})
        if traj not in table:
            raise ConfigError(f"{s['id']}: unknown trajectory id {traj!r}")
        s["trajectory_id"] = traj
        s["trajectory"] = copy.deepcopy(table[traj])
    elif isinstance(traj, dict):
        s.setdefault("trajectory_id", None)
    else:
        raise ConfigError(f"{s['id']}: missing trajectory")
    mode = s["fire"].get("mode")
    if isinstance(mode, bool):  # YAML 1.1 parses unquoted off/on as booleans
        s["fire"]["mode"] = "static" if mode else "off"
    if s["trajectory"].get("type") not in TRAJECTORY_TYPES:
        raise ConfigError(f"{s['id']}: trajectory type must be one of {TRAJECTORY_TYPES}")
    if s["fire"].get("mode") not in FIRE_MODES:
        raise ConfigError(f"{s['id']}: fire.mode must be one of {FIRE_MODES}")
    kits = s["fire"].get("kits") or []
    if any(k not in FIRE_KITS for k in kits):
        raise ConfigError(f"{s['id']}: fire.kits must be a subset of {FIRE_KITS}")
    if s["fire"].get("flame") is not None and s["fire"]["flame"] not in FLAME_VARIANTS:
        raise ConfigError(f"{s['id']}: fire.flame must be one of {FLAME_VARIANTS}")
    ref = s["trajectory"].get("height_ref", "center")
    if ref not in ("center", "canopy", "ground"):
        raise ConfigError(f"{s['id']}: trajectory.height_ref must be center, canopy or ground")
    if ref != "center" and "clearance" not in s["trajectory"]:
        raise ConfigError(f"{s['id']}: trajectory.height_ref {ref!r} needs trajectory.clearance (m)")
    preset = s["light"].get("preset")
    if preset is not None and preset not in SUN_PRESETS:
        raise ConfigError(f"{s['id']}: light.preset must be one of {sorted(SUN_PRESETS)}")
    unknown = [m for m in s["modalities"] if m not in MODALITY_TYPES]
    if unknown:
        raise ConfigError(f"{s['id']}: unknown modalities {unknown}")
    if "rgb" not in s["modalities"] or "depth" not in s["modalities"]:
        raise ConfigError(f"{s['id']}: modalities must include rgb and depth")
    return s


def select(cfg, only=None):
    if not only:
        return cfg["sequences"]
    want = set(only)
    seqs = [s for s in cfg["sequences"] if s["id"] in want or s.get("alias") in want]
    missing = want - {s["id"] for s in seqs} - {s.get("alias") for s in seqs}
    if missing:
        raise ConfigError(f"unknown sequences: {sorted(missing)}")
    return seqs
