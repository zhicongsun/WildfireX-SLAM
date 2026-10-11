# wildfirex

Toolkit of [WildfireX-SLAM](https://github.com/zhicongsun/WildfireX-SLAM): drive the WildfireX-Sim wildfire simulator
(Unreal Engine 5 + AirSim) and collect low-altitude UAV data in burning forests — RGB, depth, surface normals, albedo,
roughness, specular and object-ID images with exact camera poses at 30 Hz — under runtime-controlled fire, smoke, fog,
falling leaves and time of day. Every sequence of the WildfireX-SLAM v2.0 dataset can be regenerated with it.

![WildfireX-SLAM v2.0](https://raw.githubusercontent.com/zhicongsun/WildfireX-SLAM/main/images/v2_overview.jpg)

## Install

```bash
pip install wildfirex
```

The packaged simulator WildfireX-Sim (Windows, DirectX 12 GPU) is distributed separately (release coming soon).

## Use

```bash
set WILDFIREX_SIM=C:\WildfireX-Sim-1.1-Win64\ElectricDreamsSample.exe
wildfirex check-sim                      # one RGB-D frame + timing
wildfirex collect my_config.yaml         # collect sequences with QA, retries and resume
wildfirex qa      output/my_run/seq-0001
```

```python
from wildfirex.io.sequence import Sequence
seq = Sequence("WildfireX-SLAM/sequences/ed01-0002")
rgb, depth, c2w, K = seq.rgb(0), seq.depth(0), seq.poses[0], seq.K   # depth in metres, OpenCV camera-to-world
```

## Links

- Project page: https://zhicongsun.github.io/wildfirexslam
- Code, configs and documentation: https://github.com/zhicongsun/WildfireX-SLAM
- Dataset: https://huggingface.co/datasets/zhicongsun/WildfireX-SLAM
- Paper: https://arxiv.org/abs/2510.27133

## Citation

```bibtex
@inproceedings{sun2026wildfirex,
  title={WildfireX-SLAM: A Large-Scale Low-Altitude RGB-D Dataset for Wildfire SLAM and Beyond},
  author={Sun, Zhicong and Lo, Jacqueline and Hu, Jinxing},
  booktitle={International Conference on Multimedia Modeling},
  pages={533--546},
  year={2026},
  organization={Springer}
}
```

License: MIT. Unreal® is a trademark or registered trademark of Epic Games, Inc. in the United States of America and elsewhere.
