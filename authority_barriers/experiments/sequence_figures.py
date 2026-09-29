"""Sequences of consecutive Meshcat snapshots of the simulation, one fixed camera per scenario: four frames evenly
spaced over the maneuver (from the start to the stop, the contact or the closest approach), taken from a
three-quarter view behind the team, beside and above, at the distance at which the whole maneuver and the wall fit
the frame while the vehicles, cables and thrust vectors stay legible. The wall carries a 1 m grid for scale, the
payload's earlier positions of the sequence are drawn as translucent spheres, the gray line from the payload is
its distance h to the wall and the purple mark on it is the stopping point x_L + D n (the gap to the wall is H).

  python -m authority_barriers.experiments.sequence_figures                 # the headline trials (trial_figures.headline_items)
  python -m authority_barriers.experiments.sequence_figures --trial <stem>  # one trial log
Output: results/core/figures/trials/<trial>/sequence_<j>.png, one frame per file with t, h, v, D, H in a bar above the
frame; the trial's index.md gets a "Sequence" section. A paper composes the frames side by side."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from authority_barriers.experiments.make_figures import RES
from authority_barriers.experiments.trial_figures import Rec, headline_items, rec_from_log, rec_from_collocation

N_FRAMES = 4


def frame_indices(rec: Rec, n: int = N_FRAMES) -> list[int]:
    """Evenly spaced ticks from the start to the end of the maneuver: the closest approach for a trial that ends at the
    wall, the (near) stop of the payload for a trial that stops, the last tick otherwise."""
    v, h, t = rec.v, rec.h, rec.t
    stop = np.where(v <= max(0.05, 0.02 * abs(v[0])))[0]                     # the payload has (all but) stopped
    if rec.meta.get("termination") == "wall_contact":
        k_end = int(np.argmin(h))
    elif stop.size and stop[0] > 0:
        k_end = int(stop[0])
    else:
        k_end = int(np.argmin(h)) if h.min() < 0.05 else t.size - 1
    if v.max() < 1.0:                                                        # no braking scenario (transport with a drift): until the drift is damped
        k_end = max(int(np.argmax(h <= h.min() + 0.05)), 2)
    k_end = max(k_end, 2)
    return [int(round(x)) for x in np.linspace(0, k_end, n)]


def scene_points(rec: Rec, ks) -> np.ndarray:
    """Payload and vehicle positions at the frames plus the payload's projection onto the wall: what the camera must keep in view."""
    p = rec.p; pts = []
    for k in ks:
        st = rec.state(k); pts.append(st.x_L)
        for i in range(p.N):
            pi = st.x_L + p.l_arr[i] * st.q[i]; pts += [pi, pi + rec.U[k][i] / p.f_max_arr[i]]        # vehicle and the tip of its thrust rod
    walls = rec.extras.get("walls") or [p]
    xl = rec.state(ks[-1]).x_L; pw = min(walls, key=lambda w: w.d0 - w.n_vec @ xl)          # the nearest wall at the last frame
    pts.append(xl + (pw.d0 - pw.n_vec @ xl) * pw.n_vec)
    return np.array(pts)


def render_sequence(rec: Rec, out: Path, title: str = "", cam_pts: np.ndarray | None = None, n: int = N_FRAMES) -> list[str]:
    from authority_barriers.simulator.viz import SceneRenderer, stamp
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("sequence_*")): old.unlink()
    ks = frame_indices(rec, n); p = rec.p
    cfg = rec.meta.get("config") if isinstance(rec.meta.get("config"), dict) else {}
    use_plant = cfg.get("actuator") == "attitude" and rec.extras.get("x_plant") is not None
    R = SceneRenderer(p, width=1800, height=1300, walls=rec.extras.get("walls"))
    pts = scene_points(rec, ks) if cam_pts is None else cam_pts
    xl_end = rec.state(ks[-1]).x_L
    R.set_wall_grid(pts.mean(axis=0) if cam_pts is not None else xl_end, half_w=max(2.5, 0.5 * np.ptp(pts @ p.r_vec) + 2.0), half_h=max(2.0, 0.5 * np.ptp(pts[:, 2]) + 1.5))
    walls = rec.extras.get("walls") or [p]
    direction = 0.55 * p.r_vec - 0.25 * p.n_vec + 0.80 * np.array([0, 0, 1.0]) if len(walls) > 1 else None   # a corridor: from high behind, the drift lateral in the image
    cam = R.frame(pts, direction=direction)
    names, infos = [], []
    try:
        for j, k in enumerate(ks):
            ghosts = [rec.state(kk).x_L for kk in ks[:j]]
            if use_plant:
                kp = int(np.clip(np.searchsorted(rec.extras["t_plant"], rec.t[k]), 0, rec.extras["t_plant"].size - 1))
                R.show(x_plant=rec.extras["x_plant"][:, kp], u=rec.U[k], path=rec.X[0:3, :k + 1], ghosts=ghosts, D=rec.D[k], h=rec.h[k])
            else:
                st = rec.state(k); R.show(st=st, u=rec.U[k], v_L=st.v_L, path=rec.X[0:3, :k + 1], ghosts=ghosts, D=rec.D[k], h=rec.h[k])
            R.frame(pts, direction=direction)                                # the same camera for every frame
            f = out / f"sequence_{j}.png"; R.screenshot(f, settle=1.5 if j == 0 else 0.7)
            info = f"frame {j + 1}/{len(ks)}:  t = {rec.t[k]:.2f} s   h = {rec.h[k]:.2f} m   v = {rec.v[k]:.2f} m/s   D = {rec.D[k]:.2f} m   H = {rec.H[k]:.2f} m"
            names.append(f.name); infos.append(info)
    finally:
        R.close()
    from authority_barriers.simulator.viz import crop_to_content
    crop_to_content([out / nm for nm in names])                              # the same crop for every frame: no dead space, camera still fixed
    from authority_barriers.simulator.viz import stamp_size
    from PIL import Image
    texts = [info.split(":  ", 1)[1].replace("   ", "  ") for info in infos]                       # t, h, v, D, H above the frame
    size = min(stamp_size(Image.open(out / nm).width, tx) for nm, tx in zip(names, texts))           # one font size for the sequence
    for nm, tx in zip(names, texts): stamp(out / nm, tx, size=size)
    # the index.md section (replaced if present)
    idx = out / "index.md"
    body = idx.read_text() if idx.exists() else f"# {rec.name}\n\n{title}\n"
    head, _, _ = body.partition("\n## Sequence")
    sec = ["", "## Sequence (consecutive snapshots, one fixed camera)", "",
           f"Four frames evenly spaced over the maneuver from a fixed three-quarter camera (behind the team, beside and above; camera {np.round(cam['camera'], 1).tolist()}, target {np.round(cam['target'], 1).tolist()}, fov {cam['fov']:.0f}°): "
           "the wall with a 1 m grid, the payload's earlier positions as translucent spheres, the gray line from the payload its distance h to the wall, the purple mark the stopping point x_L + D n (the gap to the wall is H), colored lines the commanded thrust (1 m at f_max), black line the payload velocity, gray line the payload's path.", ""]
    sec += [f"- `{nm}`: {info}" for nm, info in zip(names, infos)]
    sec += [""]
    idx.write_text(head.rstrip("\n") + "\n" + "\n".join(sec))
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", default=None, help="trial log stem (path without .npz/.json)")
    ap.add_argument("--collocation", default=None, help="collocation job json")
    ap.add_argument("--title", default="")
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    args = ap.parse_args()
    if args.trial:
        rec = rec_from_log(Path(args.trial)); render_sequence(rec, RES / "core" / "figures" / "trials" / rec.name, args.title, n=args.frames); print("rendered", rec.name); return
    if args.collocation:
        rec = rec_from_collocation(Path(args.collocation)); render_sequence(rec, RES / "core" / "figures" / "trials" / rec.name, args.title, n=args.frames); print("rendered", rec.name); return
    items = headline_items()
    recs = [(rec_from_log(stem), title) for stem, title in items if Path(str(stem) + ".npz").exists()]
    # the E1 pair shares one camera so that the two scenarios are comparable frame by frame
    e1 = [r for r, _ in recs if r.name.startswith("e1")]
    shared = np.vstack([scene_points(r, frame_indices(r, args.frames)) for r in e1]) if len(e1) == 2 else None
    for rec, title in recs:
        render_sequence(rec, RES / "core" / "figures" / "trials" / rec.name, title, cam_pts=shared if rec.name.startswith("e1") else None, n=args.frames)
        print("rendered", rec.name, flush=True)
    job = RES / "core" / "e3_collocation_jobs" / "B_v4_k41_s6_dt0.01-0.1_reg.json"
    if job.exists():
        rec = rec_from_collocation(job)
        render_sequence(rec, RES / "core" / "figures" / "trials" / rec.name, "E3 direct collocation, configuration B at 4 m/s from h_c = 0.56 D.", n=args.frames); print("rendered", rec.name)


if __name__ == "__main__":
    main()
