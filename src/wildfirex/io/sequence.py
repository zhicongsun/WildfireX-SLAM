"""Loader for WildfireX sequences (released datasets or self-collected output).

    seq = Sequence("WildfireX-SLAM/sequences/ed01-0002")
    K = seq.K                       # 3x3 intrinsics
    c2w = seq.poses                 # (N, 4, 4) camera-to-world, OpenCV camera, NED world
    rgb, depth = seq.rgb(0), seq.depth(0)   # HxWx3 uint8 (RGB order), HxW float32 metres (0 = invalid)
"""
import json
import os

import cv2
import numpy as np


def load_intrinsics(seq_dir):
    K = {}
    with open(os.path.join(seq_dir, "intrinsics.yaml")) as f:
        for line in f:
            k, v = line.split(":")
            K[k.strip()] = float(v)
    return K


class Sequence:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.intrinsics = load_intrinsics(self.path)
        self.poses = np.loadtxt(os.path.join(self.path, "traj.txt")).reshape(-1, 4, 4)
        meta_path = os.path.join(self.path, "meta.json")
        self.meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}

    def __len__(self):
        return len(self.poses)

    @property
    def K(self):
        k = self.intrinsics
        return np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])

    @property
    def modalities(self):
        return sorted(d for d in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, d)))

    def frame_path(self, modality, k):
        return os.path.join(self.path, modality, f"{k:06d}.png")

    def rgb(self, k):
        im = cv2.imread(self.frame_path("rgb", k), cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2RGB)

    def depth(self, k):
        d = cv2.imread(self.frame_path("depth", k), cv2.IMREAD_UNCHANGED)
        return d.astype(np.float32) / self.intrinsics["depth_scale"]

    def image(self, modality, k):
        return cv2.imread(self.frame_path(modality, k), cv2.IMREAD_UNCHANGED)
