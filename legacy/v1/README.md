# WildfireX-SLAM v1.0 data-collection script

The script used for the v1.0 dataset of the MMM 2026 paper, kept unchanged for reference (git tag `v1.0`).

- `wildfirex_slam_data_collection.py` — AirSim-based collection script.
- `external-libraries/` — the Python packages it was run with (AirSim client 1.8.1, msgpack 0.5.6, msgpack-rpc-python 0.4.1,
  tornado 4.5.3), under their own licenses.

For new data use the `wildfirex` toolkit in this repository (`pip install wildfirex`). Main differences: float depth in
the same frame as RGB, exact camera poses, 30 Hz with world time advanced exactly 1/30 s per frame, runtime scene control,
automatic quality checks, and a much faster depth transfer (the pure-Python msgpack 0.5.6 used here re-parses the whole
buffer for every network chunk, ~11 s per float depth image).
