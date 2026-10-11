"""Stage a collected WildfireX-SLAM release for upload (maintainer helper, not part of the toolkit).

  python tools/stage_dataset.py <collected_dir> <dataset_dir> [config]

- hard-links every sequence that passed QA from <collected_dir>/<id> to <dataset_dir>/sequences/<id>
  (no extra disk space; sequences are never modified)
- writes <dataset_dir>/sequences.csv (one row per sequence, conditions + QA) and splits/<release>.json (benchmarks)
- copies the reproducing config and the canopy surface map (survey/surface.npz) next to the data
- copies the dataset card (docs/dataset_card.md -> README.md) and license (DATASET_LICENSE.md -> LICENSE.md)
"""
import csv
import json
import os
import shutil
import subprocess
import sys

from wildfirex import config  # pip install wildfirex


def main(src, dst, cfg_path=None):
    cfg_path = cfg_path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "wildfirex_slam_v2.yaml")
    cfg = config.load(cfg_path)
    os.makedirs(os.path.join(dst, "sequences"), exist_ok=True)
    rows, splits, missing = [], {}, []
    for s in cfg["sequences"]:
        sd = os.path.join(src, s["id"])
        qa_path = os.path.join(sd, "qa.json")
        if not os.path.exists(qa_path):
            missing.append(s["id"])
            continue
        with open(qa_path) as f:
            qa = json.load(f)
        if not qa.get("pass"):
            missing.append(s["id"] + " (QA failed)")
            continue
        with open(os.path.join(sd, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        out = os.path.join(dst, "sequences", s["id"])
        if not os.path.exists(out):
            subprocess.run(["cp", "-al", sd, out], check=True)  # NTFS hard links via Git Bash / MSYS
        sc = meta["sequence_config"]
        splits.setdefault(s.get("benchmark", "other"), []).append(s["id"])
        rows.append({
            "id": s["id"], "alias": s.get("alias"), "benchmark": s.get("benchmark"), "site": sc["fire"].get("site", "ed01"),
            "trajectory_id": s.get("trajectory_id"), "trajectory_type": sc["trajectory"]["type"],
            "height_ref": sc["trajectory"].get("height_ref", "center"), "clearance_m": sc["trajectory"].get("clearance"),
            "fire_mode": sc["fire"].get("mode"), "flame": sc["fire"].get("flame") if sc["fire"].get("mode") == "static" else "",
            "light": sc["light"].get("preset", "default"), "fog": sc["weather"].get("fog", 0.0), "leaf": sc["weather"].get("leaf", 0.0),
            "frames": meta["n_frames"], "fps": meta["fps"], "valid_depth": round(qa["valid_depth_mean"], 4),
            "reproj_rel_err_median": round(qa["reproj_rel_err_median"], 5), "simulator": meta.get("simulator_detected"),
        })
    with open(os.path.join(dst, "sequences.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.makedirs(os.path.join(dst, "splits"), exist_ok=True)
    tag = cfg["name"].lower().replace(" ", "_").replace("-", "_").replace(".", "_")
    with open(os.path.join(dst, "splits", f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"release": cfg["name"], "benchmarks": splits}, f, indent=2)
    shutil.copy(cfg_path, os.path.join(dst, os.path.basename(cfg_path)))
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for a, b in (("docs/dataset_card.md", "README.md"), ("DATASET_LICENSE.md", "LICENSE.md")):
        shutil.copy(os.path.join(repo, a), os.path.join(dst, b))
    surf = os.path.join(src, "_survey", "surface.npz")  # canopy map the paths were built from (exact reproduction)
    if os.path.exists(surf):
        os.makedirs(os.path.join(dst, "survey"), exist_ok=True)
        shutil.copy(surf, os.path.join(dst, "survey", "surface.npz"))
    print(f"{len(rows)} sequences, {sum(r['frames'] for r in rows)} frames -> {dst}")
    if missing:
        print("not staged:", missing)


if __name__ == "__main__":
    main(*sys.argv[1:])
