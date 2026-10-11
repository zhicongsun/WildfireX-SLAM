# Config schema (schema_version 1)

```yaml
schema_version: 1                 # required; bumped only on breaking changes
name: My release                  # recorded in every meta.json
output_dir: ../output/mine        # relative to this file

simulator:
  executable: null                # or a path; --sim and $WILDFIREX_SIM also work
  version: WildfireX-Sim 1.0      # free text, recorded in meta.json
  port: 41451
  settings: null                  # AirSim settings json; default = bundled resources/airsim_settings.json

defaults:                         # merged into every sequence (sequence fields win)
  speed: 3.0                      # m/s
  fps: 30
  game_speed: 0.1                 # world slow-down while one frame of world time runs
  warmup_sec: 30.0                # world seconds at normal speed before the first frame
  modalities: [rgb, depth, normal, ir, albedo, roughness, specular]   # also: segmentation, metallic
  fire: {mode: static}
  light: {}
  weather: {fog: 0.0, leaf: 0.0, dust: 0.0}
  max_frames: null

trajectories:                     # reusable; paired sequences reference the same id -> identical paths
  T01: {type: spiral, center: fire, radius: 72, turns: 0.28, phi_start_deg: 34, phi_end_deg: 34, n_points: 120}
  T02: {type: zigzag, center: fire, height: 60, width: 20, length: 20, spacing: 10, pitch_deg: -90, yaw_deg: 0}
  T03: {type: search, center: fire, start_offset: [-40, -40, 75], end_height: 60, curvature: 0.4, seed: 7}

sequences:
  - {id: ed01-0001, alias: A1_nofire_spiral, benchmark: A, trajectory: T01, fire: {mode: "off"}}
```

## Fields

| Field | Values |
|---|---|
| `fire.mode` | `off` (fire actors moved underground), `static` (level fire), `spread` (reserved for WildfireX-Sim 2.x) |
| `fire.site` | `ed01` (the level's original fire, default), a site id from `sites` / the bundled site list, or `[x, y]` NED — *Sim 1.1* |
| `fire.kits` | subset of `[A, B, C]`: movable fire kits (flame + smoke + orange light) placed on the ground at the site — *Sim 1.1* |
| `fire.flame` | flame variant, e.g. `NFire_Grd_00` (see `scene/presets.py::FLAME_VARIANTS`) — *Sim 1.1* |
| `fire.heading_deg`, `fire.spacing` | kits are laid out as a line (fire front) through the site: direction (deg from north) and spacing (m, default 12) |
| `fire.original_smoke` | keep the level's original smoke column (default: only at `ed01`) |
| `fire.glow` | fire light of the kits: `auto` (default: faint at `dusk`, stronger at `night`, off otherwise), `true` (full strength), `false`, or a number: metres the light is placed below the ground (it casts no shadows, so deeper = dimmer; `simSetLightIntensity` does not work on these lights) |
| `light` | `{}` level default · `{preset: noon \| morning \| dusk \| night}` (sun elevation 75°, 30°, 22°, below the horizon; dusk comes from the opposite azimuth to morning) · `{sun_pitch_deg: x}` (negative = above horizon), `{sun_yaw_deg: y}`; *Sim 1.1*: `night` also raises the moon (`moon: true/false` overrides) |
| `weather.fog` / `leaf` / `dust` | 0–1 (AirSim weather API) |
| `trajectory.center` | `fire` (the active fire site) or `[x, y, z]` NED; optional `offset: [dx, dy]` (moves the whole path); spiral / search: `look_at: fire` keeps the camera on the fire when the path is offset |
| `trajectory.height_ref` | `center` (default: heights relative to the site), `canopy` (follow the surveyed canopy) or `ground` (follow terrain) |
| `trajectory.clearance` | metres above canopy / ground when `height_ref` is `canopy` / `ground`; `safety_radius` (default 6 m) is checked around each point |
| `sites` (top level) | `{name: [x, y]}` extra fire sites in NED for this config |
| heights / offsets | metres relative to the center, positive = up |

`height_ref: canopy` needs a surface map: run `wildfirex survey <config>` once (writes `<output_dir>/_survey/surface.npz`
and proposes forest fire sites in `sites.json`), then `wildfirex survey <config> --refine` (re-surveys the canopy from
~40 m above it around every site of the config: from 180 m the renderer drops thin foliage and tall crowns come out
10 m+ too low, so flights clip tree tops). `collect` picks the map up automatically; pass `--surface` to use a published
map (e.g. the one shipped with WildfireX-SLAM v2.0) for exact reproduction.

## Ids

`<site>-<number>`, e.g. `ed04-0002` (`ed` = Electric Dreams map, site 04, sequence 0002). Conditions are fields, never part of the id,
so new conditions (wind, spread rate, ...) never force a rename. `alias` is a free human-readable name.
