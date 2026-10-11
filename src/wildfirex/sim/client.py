"""Matched AirSim Python client.

The WildfireX simulator ships a modified AirSim plugin (24 image types, simSetGameSpeed, ...).
The official `airsim` pip package does not know these extensions, so the matching client is
vendored in `_vendor/` together with msgpack-rpc-python and tornado 4 (see _vendor/LICENSES).

`_vendor/msgpackrpc/transport/tcp.py` is patched to work with msgpack>=1.0 (C extension).
With the old pure-Python msgpack 0.5 fallback, decoding one 640x480 float depth image
took ~11 s because the unpacker re-parsed the whole buffer on every received chunk.
"""
import os
import sys

_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

import msgpack  # noqa: E402

if msgpack.version < (1, 0, 0):
    raise ImportError(
        f"wildfirex requires msgpack>=1.0 (found {msgpack.version}); "
        "the old pure-Python msgpack makes float depth transfer ~100x slower")

import airsim  # noqa: E402,F401
