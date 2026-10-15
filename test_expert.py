import sys, numpy as np, gym_hil
from gym_hil import make_env
from scripted_expert import PickCubeExpert

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
env = make_env("gym_hil/PandaPickCubeBase-v0", use_viewer=False, use_inputs_control=False,
               reset_delay_seconds=0.0, render_mode="rgb_array", image_obs=False,
               random_block_position=True)
expert = PickCubeExpert()
ok, lens = 0, []
for ep in range(N):
    obs, _ = env.reset(seed=ep)
    expert.reset()
    for t in range(100):
        a = expert.act(env)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    success = bool(info.get("succeed", False))
    ok += success; lens.append(t + 1)
    print(f"ep {ep:2d}: success={success} steps={t+1} phase={expert.phase} block_z={env.unwrapped._data.sensor('block_pos').data[2]:.3f}")
print(f"SUCCESS {ok}/{N}  mean steps {np.mean(lens):.1f}  max {max(lens)}")
