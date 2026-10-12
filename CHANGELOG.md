# Changelog

Dataset releases (`vX.Y`) and toolkit releases (`wildfirex X.Y.Z`, PyPI) are versioned independently;
every `meta.json` records the toolkit, config-schema, simulator and dataset versions it was produced with.

## v2.0 — 2026-10 (dataset) · wildfirex 0.1.0 (toolkit) · WildfireX-Sim 1.1 (simulator)

Dataset
- 19 sequences, 16,572 frames over 6 fire sites (`ed01` … `ed06`), 7 pixel-aligned modalities, exact poses at 30 Hz.
- Canopy-following flights 12–15 m above the surveyed canopy; benchmarks A and CS1–CS4 (CS1 with 4 sequences;
  CS2 with dusk and a fire-lit night; CS4 with fog 0.35).
- `configs/wildfirex_slam_v2.yaml` regenerates every sequence; the canopy map is shipped with the dataset.

Toolkit (`src/wildfirex`, first public release)
- Simulator control through a vendored, matched AirSim client; msgpack ≥ 1.0 for ~125× faster float depth.
- Depth: float DepthPlanar stored as 16-bit cm PNG, captured in the same frame as RGB.
- Poses from `simGetCameraInfo`, exported as OpenCV camera-to-world in TUM and Replica formats.
- 30 Hz with 0.1 m per frame (waypoint densification + slerp); world time advanced exactly 1/30 s per frame via
  `simSetGameSpeed`.
- 5 pre-roll captures before frame 0 (the first capture after warm-up held stale render-target content).
- Runtime scene control: fire on/off, movable fire kits at any site, sun presets (dusk from the opposite azimuth to
  morning), moon, fire light dimmed by preset, fog, falling leaves.
- `wildfirex survey` (canopy surface map in UE world coordinates, automatic forest site selection) and
  `survey --refine` (low-altitude re-survey around fire sites); canopy-following trajectories (`height_ref`, `clearance`),
  `offset` / `look_at: fire` for paths beside the fire.
- Automatic QA (missing frames, valid depth, camera-inside-geometry, depth reprojection, first-frame consistency),
  retries, simulator restarts, resume; `Sequence` loader.

Simulator 1.1: 1.0 + three movable fire kits (flames, smoke, fire light), a moon and a night exposure range,
added to the level without changing existing actors.

## v1.0 — 2025-11

WildfireX-SLAM dataset (MMM 2026 paper) and data-collection script, now in `legacy/v1/` (git tag `v1.0`).
