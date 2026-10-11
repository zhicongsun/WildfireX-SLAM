# Architecture and how to extend the toolkit

This repository holds the WildfireX-SLAM project: the `wildfirex` toolkit (`src/wildfirex`, also on PyPI), the dataset
configs that regenerate every release, documentation and the v1.0 scripts. Datasets and the packaged simulator are
large binaries hosted on Hugging Face.

## Code layout

```
src/wildfirex/
  config.py          YAML schema (schema_version 1), defaults, validation
  geometry.py        pose conventions (AirSim NED <-> OpenCV camera), intrinsics
  cli.py             `wildfirex plan | check-sim | survey | collect | qa | index`
  sim/               simulator side
    launcher.py      start/stop the packaged UE environment, RPC connection, retries
    client.py        imports the vendored, matched AirSim client
    _vendor/         AirSim client (WildfireX fork), msgpackrpc (patched), tornado4
    resources/       AirSim settings template; per-simulator manifest (added actors, ground map); fire sites
  scene/             runtime scene control, never saved to the level
    presets.py       constants (sun / moon presets, flame variants, ...)
    controller.py    fire on/off, fire kits at sites, sun, moon, weather
    worldmap.py      UE <-> AirSim NED transform, ground map, surveyed canopy surface
  collect/
    trajectories.py  waypoint generators + constant-speed densification
    collector.py     per-sequence capture loop
    survey.py        canopy surface survey + automatic fire-site selection
  qa/checks.py       quality checks
  io/
    formats.py       TUM / Replica / associations / intrinsics / motion writers
    sequence.py      loader for released and self-collected sequences
configs/             wildfirex_slam_v2.yaml (reproduces v2.0); examples/ small configs for testing
simulator/           simulator versions, AirSim integration notes
docs/                schema, data format, naming, this file
```

## Extension points

| You want to add | Where | Notes |
|---|---|---|
| Fire spread (`fire.mode: spread`) | `scene/fire_spread.py`, hook in `scene/controller.py::set_fire` | needs simulator 2.x (spawnable fire actors); write the spread state as per-frame ground truth in `collect/collector.py` |
| New scene factor (wind, smoke density, ...) | `scene/controller.py` + an optional config field | new optional fields do **not** bump `schema_version` |
| New trajectory type | `collect/trajectories.py` + `config.TRAJECTORY_TYPES` + `collect/collector.py::build_path` | |
| New modality | `config.MODALITY_TYPES` + `sim/resources/airsim_settings.json` | |
| New map / fire site | new id prefix (`ed02`, ...) in the project's dataset config | ids are `<region>-<number>`; conditions stay in metadata |
| Export format (nerfstudio, COLMAP, ...) | `io/formats.py` + a `wildfirex export` command | |
| Evaluation metrics | `src/wildfirex/eval/` | benchmark protocols and baselines go to `benchmarks/` |
| New dataset release / task | a new config in `configs/` + a release tag | if a separate project grows out of it, the toolkit can move to its own repository (package name unchanged) |
| Linux headless simulator | `simulator/` docs; `sim/launcher.py` already has a non-Windows path | needs a Linux package of the UE project |

## Roadmap

1. 0.1.0 (now): simulator 1.1 (movable fire kits, moon, canopy survey, canopy-following flights), used for WildfireX-SLAM v2.0
2. Simulator 2.0: fire spread with per-frame fire-front ground truth; smoke / fire masks; geometry-only depth
3. Physical thermal rendering; near-ground / UGV views; more maps; Linux headless build
4. Shared evaluation utilities (`wildfirex.eval`)
