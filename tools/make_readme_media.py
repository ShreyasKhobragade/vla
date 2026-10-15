"""Build the figures used in README.md from the files in eval_results/.

  python tools/make_readme_media.py

Writes to assets/:
  rollouts.gif                  pick-cube: expert, ACT and SmolVLA on the same cube (seed 1000)
  failure_act.gif               pick-cube: an ACT failure (seed 1001), the gripper closes too early
  success_curve_{light,dark}.png             pick-cube success rate vs training step
  pickplace_rollouts.gif        pick-and-place: expert, ACT and SmolVLA on the same cube and pad (seed 1011)
  pickplace_compare.gif         pick-and-place: ACT stalls short of the pad, SmolVLA places it (seed 1002)
  pickplace_curve_{light,dark}.png           pick-and-place success rate vs training step
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval_results"
ASSETS = ROOT / "assets"
SCALE = 2          # upscale factor for the 128x128 camera images
HOLD_FRAMES = 8    # extra copies of the last frame so the loop pauses on the outcome
FRAME_MS = 150


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def load_rollout(name):
    """Frames from an eval GIF as (front, wrist) pairs at the original 128x128 resolution."""
    gif = Image.open(RESULTS / f"{name}.gif")
    frames = []
    for i in range(gif.n_frames):
        gif.seek(i)
        im = gif.convert("RGB").resize((256, 128), Image.NEAREST)  # eval GIFs are saved at 3x
        frames.append((im.crop((0, 0, 128, 128)), im.crop((128, 0, 256, 128))))
    return frames


def up(im):
    return im.resize((im.width * SCALE, im.height * SCALE), Image.NEAREST)


def make_rollouts(columns, out_name, max_frames=None):
    """Side-by-side rollouts: one column per (title, eval GIF name), front camera above wrist camera."""
    rollouts = [load_rollout(f)[:max_frames] for _, f in columns]
    n = max(len(r) for r in rollouts) + HOLD_FRAMES
    cell, header, gap = 128 * SCALE, 34, 6
    width = len(columns) * cell + (len(columns) - 1) * gap
    height = header + 2 * cell + gap
    bg = (13, 17, 23)
    out = []
    for t in range(n):
        canvas = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(canvas)
        for c, ((title, _), frames) in enumerate(zip(columns, rollouts)):
            front, wrist = frames[min(t, len(frames) - 1)]
            x = c * (cell + gap)
            draw.text((x + cell // 2, header // 2), title, fill=(240, 239, 236), font=font(15, True), anchor="mm")
            canvas.paste(up(front), (x, header))
            canvas.paste(up(wrist), (x, header + cell + gap))
        out.append(canvas)
    save_gif(out, ASSETS / out_name)


def make_failure():
    frames = load_rollout("actv2_30k_videos_seed1001_fail")[:34]
    captions = [(0, "approach"), (11, "gripper starts closing 5 cm above the cube"),
                (17, "closed on empty air"), (22, "lifts nothing")]
    cell, footer = 128 * SCALE, 34
    out = []
    for t in range(len(frames) + HOLD_FRAMES):
        front, wrist = frames[min(t, len(frames) - 1)]
        canvas = Image.new("RGB", (2 * cell, cell + footer), (13, 17, 23))
        canvas.paste(up(front), (0, 0))
        canvas.paste(up(wrist), (cell, 0))
        text = [c for s, c in captions if t >= s][-1]
        ImageDraw.Draw(canvas).text((cell, cell + footer // 2), f"ACT, seed 1001: {text}",
                                    fill=(240, 239, 236), font=font(13), anchor="mm")
        out.append(canvas)
    save_gif(out, ASSETS / "failure_act.gif")


def save_gif(frames, path):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=FRAME_MS, loop=0, optimize=True)
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def curve(prefix):
    steps, rates, lo, hi = [], [], [], []
    for s in range(5000, 30001, 5000):
        rows = list(csv.DictReader(open(RESULTS / f"{prefix}_{s:06d}.csv")))
        k, n = sum(int(r["success"]) for r in rows), len(rows)
        a, b = wilson(k, n)
        steps.append(s / 1000), rates.append(100 * k / n), lo.append(100 * a), hi.append(100 * b)
    return np.array(steps), np.array(rates), np.array(lo), np.array(hi)


THEMES = {
    "light": dict(surface="#ffffff", text="#0b0b0b", muted="#52514e", grid="#e4e3df",
                  series=["#2a78d6", "#eb6834"]),
    "dark": dict(surface="#0d1117", text="#f0efec", muted="#c3c2b7", grid="#2b2d31",
                 series=["#3987e5", "#d95926"]),
}


def make_curve(theme, prefixes, title, out_name, y_min):
    th = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor(th["surface"])
    ax.set_facecolor(th["surface"])
    series = [("ACT", curve(prefixes[0])), ("SmolVLA", curve(prefixes[1]))]
    handles = []
    for (label, (x, y, lo, hi)), color, dx, dy in zip(series, th["series"], (-0.25, 0.25), (7, -7)):
        ax.errorbar(x + dx, y, yerr=[y - lo, hi - y], color=color, lw=2, marker="o", ms=6,
                    elinewidth=1.2, capsize=0, label=label, zorder=3,
                    markeredgecolor=th["surface"], markeredgewidth=1.5)
        handles.append(plt.Line2D([], [], color=color, lw=2, marker="o", ms=6, label=label))
        ax.annotate(f"{label}  {y[-1]:.0f}%", (x[-1] + dx, y[-1]), xytext=(10, dy),
                    textcoords="offset points", va="center", color=th["text"], fontsize=10)
    ax.axhline(100, color=th["muted"], lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(5, 101.5, "scripted expert 100%", color=th["muted"], fontsize=9, va="bottom")
    ax.set_xlim(3.5, 34)
    ax.set_ylim(y_min, 106)
    ax.set_xticks(range(5, 31, 5), [f"{s}k" for s in range(5, 31, 5)])
    tick = 10 if y_min >= 50 else 20
    ax.set_yticks(range(y_min, 101, tick), [f"{v}%" for v in range(y_min, 101, tick)])
    ax.set_xlabel("training steps", color=th["muted"])
    ax.set_title(title, color=th["text"], fontsize=11, loc="left", pad=12)
    ax.grid(axis="y", color=th["grid"], lw=0.8)
    ax.tick_params(colors=th["muted"], length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(th["grid"])
    leg = ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(th["text"])
    fig.text(0.125, 0.01, "Bars: 95% Wilson confidence intervals. Points nudged sideways so the bars don't overlap.",
             color=th["muted"], fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = ASSETS / f"{out_name}_{theme}.png"
    fig.savefig(path, facecolor=th["surface"])
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    make_rollouts([("Scripted expert", "expert_seed1000_ok"),
                   ("ACT (30k steps)", "actv2_30k_videos_seed1000_ok"),
                   ("SmolVLA (30k steps)", "smolvla30k_030000_seed1000_ok")], "rollouts.gif")
    make_failure()
    make_rollouts([("Scripted expert", "pickplace_expert_seed1011_ok"),
                   ("ACT (30k steps)", "pp_act_030000_seed1011_ok"),
                   ("SmolVLA (30k steps)", "pp_smolvla_030000_seed1011_ok")], "pickplace_rollouts.gif")
    make_rollouts([("ACT: stalls short of the pad", "pp_act_030000_seed1002_fail"),
                   ("SmolVLA: places the cube", "pp_smolvla_030000_seed1002_ok")],
                  "pickplace_compare.gif", max_frames=80)
    for theme in THEMES:
        make_curve(theme, ("actv2", "smolvla30k"),
                   "Pick cube: success on 50 fixed test positions (cube lifted ≥ 5 cm)", "success_curve", 50)
        make_curve(theme, ("pp_act", "pp_smolvla"),
                   "Pick and place: success on 50 fixed test positions (cube left on the pad)", "pickplace_curve", 0)
