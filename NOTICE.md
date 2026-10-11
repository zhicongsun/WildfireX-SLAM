# Third-party components and acknowledgements

## Vendored code (`src/wildfirex/sim/_vendor/`)

| Component | Version | License | Changes |
|---|---|---|---|
| AirSim Python client, WildfireX fork (matches the AirSim-WildfireX plugin) | 1.7.0-wildfirex | MIT (`LICENSES/AirSim-LICENSE.txt`) | extended `ImageType`, extra RPCs such as `simSetGameSpeed` |
| msgpack-rpc-python | 0.4.1 | Apache-2.0 | `transport/tcp.py`: compatibility with msgpack ≥ 1.0 |
| tornado (renamed `tornado4`) | 4.5.3 | Apache-2.0 (`LICENSES/tornado-LICENSE.txt`) | none (test suite removed) |

msgpack ≥ 1.0 (C extension) is required: with the pure-Python msgpack 0.5 fallback, the RPC unpacker
re-parses the whole buffer for every received network chunk and one 640×480 float depth image took ~11 s.

## Simulator

The AirSim-WildfireX plugin is a modified version of [Microsoft AirSim](https://github.com/microsoft/AirSim)
(MIT) with extended G-buffer image types and `simSetGameSpeed`, building on the work of community forks such
as [Cosys-AirSim](https://github.com/Cosys-Lab/Cosys-AirSim).

The packaged simulator contains Epic Games sample content (Electric Dreams Environment), Quixel Megascans
assets and the M5 VFX Vol.2 Niagara fire pack. They are distributed only in packaged form under their
respective licenses; source assets are not redistributed.

## Acknowledgements

This work builds on [AirSim](https://github.com/microsoft/AirSim),
[Cosys-AirSim](https://github.com/Cosys-Lab/Cosys-AirSim),
[ProjectAirSim](https://github.com/iamaisim/ProjectAirSim),
[PromptCraft-Robotics](https://github.com/microsoft/PromptCraft-Robotics) and
[airsim_agent](https://github.com/maris205/airsim_agent). We thank the authors.

## v1.0 scripts

`legacy/v1/external-libraries/` contains the third-party Python packages used by the v1.0 script (AirSim client 1.8.1,
msgpack 0.5.6, msgpack-rpc-python 0.4.1, tornado 4.5.3) under their respective licenses.

## Unreal Engine

WildfireX-Sim uses Unreal® Engine. Unreal® is a trademark or registered trademark of Epic Games, Inc. in the United States of
America and elsewhere. The dataset license ([DATASET_LICENSE.md](DATASET_LICENSE.md)) passes on the "NoAI" terms of the
third-party assets depicted in the data.
