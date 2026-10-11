# WildfireX-SLAM Dataset License

Copyright (c) 2025-2026 the WildfireX-SLAM authors (Zhicong Sun, Jacqueline Lo, Jinxing Hu).

The WildfireX-SLAM dataset (images, depth and other modalities, poses, metadata and annotations) is licensed under the
terms of the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**
(<https://creativecommons.org/licenses/by/4.0/legalcode>), **with one additional restriction** stated below.
Because of this restriction, this license is based on, but is not identical to, CC BY 4.0; Creative Commons is not
affiliated with it.

## 1. What you may do

Under the CC BY 4.0 terms you may copy, redistribute, adapt and build upon the dataset for any purpose, including
commercially, as long as you give appropriate credit (cite the paper below), provide a link to this license and
indicate if changes were made.

## 2. Additional restriction: no generative AI

You may not use the dataset, in whole or in part:

1. in datasets utilized by Generative AI Programs;
2. in the development of Generative AI Programs; or
3. as training inputs to Generative AI Programs.

"Generative AI Programs" means artificial intelligence, machine learning, deep learning, neural networks, or similar
technologies designed to automate the generation of or aid in the creation of new content, including but not limited
to audio, visual, or text-based content. Programs do not meet this definition where they, for example, (a) solely
operate on the original content; (b) generate tags to classify visual input content; or (c) generate instructions to
arrange existing content, without creating new content.

This restriction passes on the "NoAI" terms of the third-party assets depicted in the data (section 3). It does not
affect the intended uses of the dataset, for example SLAM and visual odometry, pose and depth estimation, 3D
reconstruction and rendering of the captured scenes, detection, segmentation, and benchmarking of such methods.

If you redistribute the dataset or an adaptation of it, you must pass on this restriction.

## 3. Third-party content

The data were rendered in Unreal Engine from third-party assets, which remain the property of their licensors:

- *Electric Dreams Environment* sample (Epic Games, Inc.), including the Quixel Megascans assets shipped with it,
  licensed under the Epic Content License Agreement as UE-Only Content;
- *M5 VFX Vol2. Fire and Flames (Niagara)* (M5VFX), licensed through the Epic marketplace.

Both are tagged "NoAI" by their licensors. The license terms of these assets permit the distribution of rendered images
and video. This license grants no rights to the underlying assets themselves.

Unreal® is a trademark or registered trademark of Epic Games, Inc. in the United States of America and elsewhere.

## 4. Citation

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

## 5. No warranty

The dataset is provided "as is", without warranties of any kind, as set out in Section 5 of CC BY 4.0.
