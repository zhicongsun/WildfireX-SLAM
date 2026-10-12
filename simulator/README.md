# WildfireX-Sim (simulator)

The simulator is a packaged Unreal Engine 5.2 application. It is distributed as a binary download, not in
this repository.

| Version | Content | Download | Used by |
|---|---|---|---|
| 1.0 | Electric Dreams environment, M5 VFX Niagara fire, AirSim-WildfireX plugin (environment of the MMM 2026 paper) | `WildfireX-Sim-1.0-Win64.zip` (≈13 GB) — coming soon | WildfireX-SLAM v1.0 |
| 1.1 | 1.0 + movable fire kits (flame variants, smoke, fire light), moon, night auto-exposure range; same code | `WildfireX-Sim-1.1-Win64.zip` (≈13 GB) — coming soon | WildfireX-SLAM v2.0 |
| 2.0 | planned: dynamic fire scenes with per-frame fire-front ground truth, fire/smoke masks | – | – |

### 1.1 changes (add-only)

29 actors were added to `/Game/Levels/ElectricDreams_Env`; no existing actor or the map package was modified
(checked by MD5 over all 15,247 level files). All are always-loaded and listed with their names in
`src/wildfirex/sim/resources/wildfirex_sim_1.1.json`:

- fire kits A, B, C: flame Niagara variants (`/Game/M5VFXVOL2/Niagara/Fire/*`, scale 6), the two smoke systems used by the
  original fire, and an orange point light (400k lm, 60 m radius); parked 2 km underground until the toolkit moves them
- `WFX_moon`: DirectionalLight, 0.6 lux, bluish, atmosphere light index 1, parked below the horizon
- `WFX_ppv_night_exposure`: unbound PostProcessVolume (priority 20) overriding only `AutoExposureMinBrightness = -6`
  so night scenes can adapt; day exposure is unaffected

Requirements: Windows 10/11, DirectX 12 GPU (tested on RTX 4080 Laptop, 12 GB VRAM), ≥ 16 GB RAM.

## How AirSim is integrated

The UE project (`WildfireSLAM.uproject`) contains the AirSim plugin as a **project plugin**
(`Plugins/AirSim`), enables it in the `.uproject`, adds it as a dependency of the game module, and sets
`GlobalDefaultGameMode=/Script/AirSim.AirSimGameMode` in `Config/DefaultEngine.ini`. At startup the game mode
reads the AirSim settings (passed with `-settings=<file>` by the toolkit), spawns the ComputerVision vehicle
and opens the RPC server on port 41451. The plugin is compiled into the packaged executable.

## AirSim-WildfireX plugin

A modified version of Microsoft AirSim (MIT) with extended G-buffer image types (Velocity, ShaderID,
TextureUV, SceneDepth, ReflectionVector, DotSurfaceReflection, Metallic, Albedo, Specular, Opacity,
Roughness, Anisotropy, AO, ShadingModelColor) and `simSetGameSpeed`. It builds on the work of community
forks such as Cosys-AirSim. The matching Python client is vendored in `src/wildfirex/sim/_vendor/airsim`.
AirSim's MIT license notice ships with the simulator.

## Notes measured on 1.0

- ComputerVision mode: `ClockSpeed` does not slow the UE world and `simContinueForTime` is not implemented;
  the toolkit uses `simSetGameSpeed` + pause/resume to advance exactly 1/fps per frame.
- Exposure is locked by the level's post-process volumes (`AutoExposureBias`, `r.ExposureOffset` have no
  effect on capture cameras); `simSetTimeOfDay` has no effect — the toolkit rotates the DirectionalLight.
- Image requests wait for the game viewport to draw; a minimised window or locked screen stalls capture.

## Notes on 1.1

- The AirSim NED origin (PlayerStart) differs from 1.0: in 1.1 the original fire is at NED ≈ (26.8, 111.8, 26.5) m,
  in 1.0 at ≈ (32.4, −62.6, 36.0) m. The toolkit therefore stores world positions (sites, surface map, ground map)
  in UE world centimetres and converts at runtime; every `meta.json` records `ned_origin_ue_cm`.
- The fire kits, moon and night post-process volume are parked (2 km underground / below the horizon) at startup
  and only placed by the toolkit, so a 1.1 build without the toolkit looks exactly like 1.0.
- The moon is deliberately weak: night scenes without fire are almost black; night sequences rely on the fire light.
- New actors must be cooked with a **full** cook (`BuildCookRun -cook` without `-iterate`): an iterative cook does not
  regenerate World Partition cells when only external actor files changed.
