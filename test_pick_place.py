"""Success rate of the scripted pick-and-place expert: python test_pick_place.py [episodes] [first_seed]"""
import sys

import numpy as np

from pick_place_env import make_pick_place_env
from scripted_expert import PickPlaceExpert

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
first = int(sys.argv[2]) if len(sys.argv) > 2 else 0
env = make_pick_place_env(image_obs=False)
expert = PickPlaceExpert()
ok, lens = 0, []
for seed in range(first, first + N):
    np.random.seed(seed)
    env.reset(seed=seed)
    expert.reset()
    info = {}
    for t in range(200):
        obs, r, term, trunc, info = env.step(expert.act(env))
        if term or trunc:
            break
    u = env.unwrapped
    block, pad = u._data.sensor("block_pos").data, u.pad_pos
    success = bool(info.get("succeed", False))
    ok += success
    lens.append(t + 1)
    print(f"seed {seed}: success={success} steps={t + 1:3d} phase={expert.phase:8s} "
          f"cube-pad offset=({100 * (block[0] - pad[0]):+.1f}, {100 * (block[1] - pad[1]):+.1f}) cm  cube_z={block[2]:.3f}")
print(f"SUCCESS {ok}/{N}  mean steps {np.mean(lens):.1f}  max {max(lens)}")
