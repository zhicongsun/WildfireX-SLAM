"""Command line interface: `wildfirex <command>`.

  plan       CONFIG                 frames / path length per sequence (no simulator needed)
  check-sim  [--sim EXE]            launch the simulator, grab one RGB-D frame, report timing
  collect    CONFIG [--only ID ...] collect sequences, QA each, retry failures, resume-safe
  survey     [CONFIG]               (Sim 1.1+) canopy surface map + fire-site proposal
  qa         SEQ_DIR [...]          (re)run quality checks
  index      DATA_DIR               write sequences.csv from every meta.json / qa.json
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback

from . import __version__

PATH_FAILS = {"near_frame_ratio", "valid_depth_mean"}  # deterministic for a given path: retrying does not help


def _sim_path(args, cfg=None):
    return args.sim or (cfg or {}).get("simulator", {}).get("executable")


def cmd_plan(args):
    from . import config
    from .collect import trajectories as T
    from .collect.collector import build_path
    import numpy as np
    cfg = config.load(args.config)
    total = 0
    print(f"{'id':<12} {'alias':<34} {'traj':<5} {'frames':>6} {'path_m':>7}")
    for s in config.select(cfg, args.only):
        s = dict(s, trajectory=dict(s["trajectory"], height_ref="center"))  # surface following needs the simulator
        pts, _ = build_path(s, np.zeros(3))
        total += len(pts)
        print(f"{s['id']:<12} {str(s.get('alias', '')):<34} {str(s.get('trajectory_id', '')):<5} {len(pts):>6} {T.path_length(pts):>7.1f}")
    print(f"total frames: {total}  (~{total * 0.75 / 3600:.1f} h at ~0.75 s/frame, plus warm-up)")


def cmd_check_sim(args):
    import numpy as np
    from .collect.collector import CAMERA, VEHICLE
    from .scene.controller import Scene
    from .sim.client import airsim
    from .sim.launcher import SETTINGS_TEMPLATE, Simulator
    with Simulator(args.sim, settings=args.settings or SETTINGS_TEMPLATE, port=args.port) as sim:
        c = sim.client
        sc = Scene(c, 0.1)
        print(f"simulator: {sc.version}\nfire actors: {sc.fires}\nsun: {sc.sun}  moon: {sc.moon}\n"
              f"fire position (NED): {None if sc.fire_pos is None else sc.fire_pos.round(2).tolist()}")
        reqs = [airsim.ImageRequest(CAMERA, 0, pixels_as_float=False, compress=False),
                airsim.ImageRequest(CAMERA, 1, pixels_as_float=True, compress=False)]
        ts = []
        for _ in range(5):
            t0 = time.time()
            rs = c.simGetImages(reqs, VEHICLE)
            ts.append(time.time() - t0)
        d = np.array(rs[1].image_data_float, np.float32)
        print(f"RGB-D request: median {np.median(ts):.3f} s (expect < 0.3 s); depth median {np.median(d):.1f} m")
        ok = sc.fire_pos is not None and sc.sun is not None and np.median(ts) < 1.0
        print("check-sim:", "OK" if ok else "PROBLEM")
        return 0 if ok else 1


def cmd_collect(args):
    from . import config
    from .collect.collector import Collector
    from .qa.checks import check
    from .sim.launcher import SETTINGS_TEMPLATE, Simulator
    cfg = config.load(args.config)
    out_root = os.path.abspath(args.out or cfg["output_dir"])
    os.makedirs(out_root, exist_ok=True)
    seqs = config.select(cfg, args.only)
    if args.max_frames:
        for s in seqs:
            s["max_frames"] = args.max_frames
    settings = args.settings or cfg["simulator"].get("settings") or SETTINGS_TEMPLATE
    sim = Simulator(_sim_path(args, cfg), settings=settings, port=cfg["simulator"].get("port", 41451))
    surface = args.surface or os.path.join(out_root, "_survey", "surface.npz")
    run_info = {"settings": sim.settings, "executable": sim.executable,
                "simulator_version": cfg["simulator"].get("version"), "release": cfg.get("name"),
                "surface": surface if os.path.exists(surface) else None, "sites": cfg.get("sites", {})}
    log_path = os.path.join(out_root, "collect_log.jsonl")
    since_restart = 0
    try:
        for s in seqs:
            out = os.path.join(out_root, s["id"])
            qa_path = os.path.join(out, "qa.json")
            if not args.force and os.path.exists(qa_path):
                with open(qa_path) as f:
                    if json.load(f).get("pass"):
                        print(f"[skip] {s['id']} already passed QA")
                        continue
            for attempt in range(args.retries + 1):
                if sim.client is None or since_restart >= args.restart_every:
                    sim.stop()
                    sim.start()
                    since_restart = 0
                rec = {"id": s["id"], "alias": s.get("alias"), "attempt": attempt, "start": time.strftime("%Y-%m-%d %H:%M:%S")}
                try:
                    t0 = time.time()
                    Collector(s, sim.client, out_root, run_info).run()
                    rec["minutes"] = round((time.time() - t0) / 60, 1)
                    rec["qa"] = check(out)
                except Exception as e:
                    rec["error"] = repr(e)
                    rec["trace"] = traceback.format_exc()[-1500:]
                    sim.stop()
                since_restart += 1
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ok = rec.get("qa", {}).get("pass")
                print(f"[{'PASS' if ok else 'FAIL'}] {s['id']} {rec.get('minutes')} min {rec.get('qa', {}).get('fails', rec.get('error'))}", flush=True)
                if ok:
                    break
                if "qa" in rec and set(rec["qa"]["fails"]) <= PATH_FAILS:
                    print(f"  {s['id']}: path-dependent QA failure, a retry would fly the same path; fix the trajectory", flush=True)
                    break
    finally:
        sim.stop()
    return 0


def cmd_survey(args):
    from . import config
    from .collect.survey import pick_sites, survey
    from .scene.controller import Scene
    from .sim.launcher import SETTINGS_TEMPLATE, Simulator
    cfg = config.load(args.config) if args.config else {"simulator": {}, "output_dir": os.path.abspath("output")}
    out_dir = os.path.join(os.path.abspath(args.out or cfg["output_dir"]), "_survey")
    os.makedirs(out_dir, exist_ok=True)
    surface_path = os.path.join(out_dir, "surface.npz")
    with Simulator(_sim_path(args, cfg), settings=args.settings or SETTINGS_TEMPLATE) as sim:
        sc = Scene(sim.client, 0.1)
        if sc.world is None:
            raise RuntimeError("survey needs WildfireX-Sim 1.1 or later")
        if args.refine:
            from .collect.survey import refine
            if not args.config or not os.path.exists(surface_path):
                raise RuntimeError("--refine needs CONFIG and an existing surface map from `wildfirex survey`")
            sc.sites.update(cfg.get("sites", {}))
            sites = sorted({s["fire"].get("site") or "ed01" for s in cfg["sequences"]})
            n = refine(sim.client, sc, surface_path, [sc.site_xy(s) for s in sites], half=args.refine_half)
            print(f"refined around {sites}: {n} cells raised -> {surface_path}")
            return 0
        (x0, x1), (y0, y1) = sc.world.bounds_ned()
        m = args.margin
        path = survey(sim.client, sc, (x0 + m, x1 - m), (y0 + m, y1 - m), os.path.join(out_dir, "surface.npz"),
                      altitude=args.altitude, spacing=args.spacing, cell=args.cell)
        sc.world.load_surface(path)
        sites = pick_sites(sc.world, n=args.sites, exclude=[sc.fire_pos[:2]])
    named = {f"ed{i + 2:02d}": s for i, s in enumerate(sites)}
    with open(os.path.join(out_dir, "sites.json"), "w", encoding="utf-8") as f:
        json.dump(named, f, indent=2)
    print(f"surface -> {path}")
    print(f"sites -> {os.path.join(out_dir, 'sites.json')}: {list(named)}")
    return 0


def cmd_qa(args):
    from .qa.checks import check
    bad = 0
    for d in args.seq_dirs:
        r = check(d)
        bad += not r["pass"]
        print(json.dumps(r))
    return 1 if bad else 0


def cmd_index(args):
    rows = []
    for name in sorted(os.listdir(args.data_dir)):
        d = os.path.join(args.data_dir, name)
        mp = os.path.join(d, "meta.json")
        if not os.path.isfile(mp):
            continue
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
        q = {}
        if os.path.isfile(os.path.join(d, "qa.json")):
            with open(os.path.join(d, "qa.json")) as f:
                q = json.load(f)
        sc = m.get("sequence_config", {})
        rows.append({
            "id": m.get("id", name), "alias": m.get("alias"), "benchmark": m.get("benchmark"),
            "trajectory_id": m.get("trajectory_id"), "trajectory_type": sc.get("trajectory", {}).get("type"),
            "fire_mode": sc.get("fire", {}).get("mode"), "light": json.dumps(sc.get("light", {})),
            "fog": sc.get("weather", {}).get("fog", 0), "leaf": sc.get("weather", {}).get("leaf", 0),
            "frames": m.get("n_frames"), "fps": m.get("fps"), "qa_pass": q.get("pass"),
            "reproj_rel_err_median": q.get("reproj_rel_err_median"),
        })
    out = args.output or os.path.join(args.data_dir, "sequences.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} sequences -> {out}")
    return 0


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="wildfirex", description=f"WildfireX toolkit {__version__}")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="frames and path length per sequence")
    p.add_argument("config")
    p.add_argument("--only", nargs="*")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("check-sim", help="launch the simulator and grab one RGB-D frame")
    p.add_argument("--sim", help="simulator executable (default: $WILDFIREX_SIM)")
    p.add_argument("--settings")
    p.add_argument("--port", type=int, default=41451)
    p.set_defaults(fn=cmd_check_sim)

    p = sub.add_parser("collect", help="collect sequences from a config")
    p.add_argument("config")
    p.add_argument("--only", nargs="*", help="sequence ids or aliases")
    p.add_argument("--sim", help="simulator executable (default: config simulator.executable or $WILDFIREX_SIM)")
    p.add_argument("--settings", help="AirSim settings json (default: bundled template)")
    p.add_argument("--out", help="output directory (default: config output_dir)")
    p.add_argument("--max-frames", type=int, help="truncate every sequence (quick tests)")
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--restart-every", type=int, default=4, help="restart the simulator every N sequences")
    p.add_argument("--force", action="store_true", help="re-collect sequences that already passed QA")
    p.add_argument("--surface", help="canopy surface .npz from `wildfirex survey` (default: <out>/_survey/surface.npz)")
    p.set_defaults(fn=cmd_collect)

    p = sub.add_parser("survey", help="(Sim 1.1+) canopy surface survey + automatic fire-site selection")
    p.add_argument("config", nargs="?")
    p.add_argument("--sim")
    p.add_argument("--settings")
    p.add_argument("--out")
    p.add_argument("--altitude", type=float, default=180.0)
    p.add_argument("--spacing", type=float, default=200.0)
    p.add_argument("--cell", type=float, default=2.0)
    p.add_argument("--margin", type=float, default=100.0)
    p.add_argument("--sites", type=int, default=5)
    p.add_argument("--refine", action="store_true",
                   help="re-survey the canopy from low altitude around every fire site of CONFIG (run after the survey)")
    p.add_argument("--refine-half", type=float, default=90.0, help="half size (m) of the refined square per site")
    p.set_defaults(fn=cmd_survey)

    p = sub.add_parser("qa", help="run quality checks on sequence directories")
    p.add_argument("seq_dirs", nargs="+")
    p.set_defaults(fn=cmd_qa)

    p = sub.add_parser("index", help="write sequences.csv for a data directory")
    p.add_argument("data_dir")
    p.add_argument("--output")
    p.set_defaults(fn=cmd_index)

    args = ap.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
