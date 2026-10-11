"""Launching / stopping the packaged simulator and connecting to its AirSim RPC server."""
import os
import socket
import subprocess
import sys
import time

from .client import airsim

DEFAULT_PORT = 41451
RPC_TIMEOUT = 90  # s; an image request hangs forever if the game viewport stops drawing
SETTINGS_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "airsim_settings.json")


def resolve_executable(path=None):
    """Simulator executable from argument, $WILDFIREX_SIM, or error."""
    path = path or os.environ.get("WILDFIREX_SIM")
    if not path:
        raise RuntimeError("Simulator executable not given: pass --sim PATH or set the WILDFIREX_SIM environment variable")
    path = os.path.expandvars(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Simulator executable not found: {path}")
    return path


def port_open(port=DEFAULT_PORT):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def connect(port=DEFAULT_PORT, timeout=RPC_TIMEOUT):
    c = airsim.VehicleClient(port=port, timeout_value=timeout)
    c.confirmConnection()
    return c


def _image_name(executable):
    """Name of the process that actually renders (packaged builds start a -Shipping child)."""
    stem = os.path.splitext(os.path.basename(executable))[0]
    return stem if stem.endswith("-Shipping") else f"{stem}-Win64-Shipping"


def _running(image):
    if sys.platform != "win32":
        r = subprocess.run(["pgrep", "-f", image], capture_output=True, text=True)
        return bool(r.stdout.strip())
    r = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {image}.exe"], capture_output=True, text=True)
    return image.lower() in r.stdout.lower()


def _error_dialog(image):
    """Packaged UE shows a window titled 'Message' on fatal startup errors (e.g. D3D12 init failure)."""
    if sys.platform != "win32":
        return False
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"Get-Process -Name '{image}' -ErrorAction SilentlyContinue | ForEach-Object {{ $_.MainWindowTitle }}"],
                       capture_output=True, text=True)
    return "Message" in r.stdout


class Simulator:
    """Context manager: start the simulator, wait for RPC, stop it on exit."""

    def __init__(self, executable, settings=SETTINGS_TEMPLATE, port=DEFAULT_PORT, resolution=(1280, 720),
                 startup_timeout=300, attempts=3, settle=20.0, extra_args=()):
        self.executable = resolve_executable(executable)
        self.settings = os.path.abspath(settings)
        self.port = port
        self.resolution = resolution
        self.startup_timeout = startup_timeout
        self.attempts = attempts
        self.settle = settle
        self.extra_args = list(extra_args)
        self.image = _image_name(self.executable)
        self.proc = None
        self.client = None

    def start(self):
        if port_open(self.port):
            raise RuntimeError(f"Port {self.port} is already in use: another AirSim/UE instance is running")
        args = [self.executable, "-windowed", f"-ResX={self.resolution[0]}", f"-ResY={self.resolution[1]}",
                f"-settings={self.settings}", "-nosound", *self.extra_args]
        last = None
        for attempt in range(self.attempts):
            self.proc = subprocess.Popen(args)
            t0 = time.time()
            while time.time() - t0 < self.startup_timeout:
                if port_open(self.port):
                    self.client = connect(self.port)
                    time.sleep(self.settle)  # let world-partition streaming settle
                    return self.client
                if _error_dialog(self.image):
                    last = "startup error dialog (e.g. DirectX 12 initialisation failed)"
                    break
                if self.proc.poll() is not None and not _running(self.image):
                    last = f"process exited with code {self.proc.returncode}"
                    break
                time.sleep(3)
            else:
                last = "timed out waiting for the AirSim RPC server"
            print(f"[wildfirex] launch attempt {attempt + 1} failed: {last}; retrying", flush=True)
            self.stop()
            time.sleep(30)
        raise RuntimeError(f"Simulator failed to start after {self.attempts} attempts: {last}")

    def stop(self, grace=15):
        if self.proc is None:
            return
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True)
            if _running(self.image):
                subprocess.run(["taskkill", "/IM", f"{self.image}.exe", "/F"], capture_output=True)
        else:
            self.proc.kill()
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pass
        t0 = time.time()
        while _running(self.image) and time.time() - t0 < 60:
            time.sleep(2)
        time.sleep(grace)  # release the GPU before the next launch
        self.proc = None
        self.client = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
