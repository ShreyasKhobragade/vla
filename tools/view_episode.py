import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from lerobot.datasets import LeRobotDataset

repo = sys.argv[1]
ep = int(sys.argv[2]) if len(sys.argv) > 2 else 0
root = Path.home() / ".cache" / "huggingface" / "lerobot" / repo
d = LeRobotDataset(repo, root=str(root))

# find the frame indices that belong to this episode
try:
    idxs = [i for i, e in enumerate(d.hf_dataset["episode_index"]) if int(e) == ep]
except Exception:
    idxs = []
    for i in range(d.num_frames):
        e = int(d[i]["episode_index"])
        if e == ep:
            idxs.append(i)
        elif idxs:
            break
if not idxs:
    sys.exit(f"episode {ep} not found in {repo}")

try:
    font = ImageFont.load_default(size=16)
except TypeError:
    font = ImageFont.load_default()

def to_img(t):
    return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")

frames = []
for n, i in enumerate(idxs):
    s = d[i]
    both = np.concatenate([to_img(s["observation.images.front"]), to_img(s["observation.images.wrist"])], axis=1)
    im = Image.fromarray(both).resize((both.shape[1] * 3, both.shape[0] * 3), Image.NEAREST)
    a = s["action"].numpy().round(2).tolist()
    ImageDraw.Draw(im).text((6, 6), f"step {n + 1}/{len(idxs)}  action {a}", fill=(255, 255, 0), font=font)
    frames.append(im)

out = f"episode_{ep}.gif"
frames[0].save(out, save_all=True, append_images=frames[1:], duration=200, loop=0)
print(f"saved {out} ({len(frames)} frames, front camera on the left, wrist camera on the right)")
