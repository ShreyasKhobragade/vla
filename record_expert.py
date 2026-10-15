"""Record scripted-expert demos for gym_hil Franka pick-cube as a LeRobotDataset.

The dataset fields match what the keyboard teleop recorder (lerobot.rl.gym_manipulator)
writes, so ACT / SmolVLA training commands work on it unchanged:
  observation.images.front / observation.images.wrist  (video, 3x128x128)
  observation.state                                     (18,)
  action                                                (4,)  [dx, dy, dz, gripper]
  next.reward, next.done, complementary_info.discrete_penalty

Only successful episodes are saved by default.

Usage (run from the folder that contains scripted_expert.py):
  python record_expert.py --repo-id local/franka_expert_v1 --episodes 50
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from gym_hil import make_env

from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD

from scripted_expert import PickCubeExpert

PENALTY_KEY = "complementary_info.discrete_penalty"


def default_root(repo_id: str) -> Path:
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME

        return Path(HF_LEROBOT_HOME) / repo_id
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id


def chw_float(img: np.ndarray) -> np.ndarray:
    """HWC uint8 -> CHW float32 in [0, 1], the same layout the teleop recorder stores."""
    return np.ascontiguousarray(np.transpose(img, (2, 0, 1)), dtype=np.float32) / 255.0


def make_frame(obs, action, reward, done, penalty, task):
    return {
        OBS_STATE: np.asarray(obs["agent_pos"], dtype=np.float32),
        f"{OBS_IMAGES}.front": chw_float(obs["pixels"]["front"]),
        f"{OBS_IMAGES}.wrist": chw_float(obs["pixels"]["wrist"]),
        ACTION: np.asarray(action, dtype=np.float32),
        REWARD: np.array([reward], dtype=np.float32),
        DONE: np.array([done], dtype=bool),
        PENALTY_KEY: np.array([penalty], dtype=np.float32),
        "task": task,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="e.g. local/franka_expert_v1 (must be new)")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--root", default=None, help="dataset folder (default: LeRobot cache)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0, help="action noise during approach, e.g. 0.1")
    ap.add_argument("--fixed-cube", action="store_true", help="cube always at the default spot")
    ap.add_argument("--task", default="pick_cube")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=100)
    args = ap.parse_args()

    root = Path(args.root) if args.root else default_root(args.repo_id)
    if root.exists():
        sys.exit(f"{root} already exists. Choose a new --repo-id (or --root), or delete that folder.")

    env = make_env(
        "gym_hil/PandaPickCubeBase-v0",
        use_viewer=False,
        use_inputs_control=False,
        reset_delay_seconds=0.0,
        gripper_penalty=-0.02,
        render_mode="rgb_array",
        image_obs=True,
        random_block_position=not args.fixed_cube,
    )
    expert = PickCubeExpert(noise=args.noise, rng=np.random.default_rng(args.seed))

    obs, _ = env.reset(seed=args.seed)
    h, w, _ = obs["pixels"]["front"].shape
    features = {
        ACTION: {"dtype": "float32", "shape": (4,), "names": ["delta_x", "delta_y", "delta_z", "gripper"]},
        REWARD: {"dtype": "float32", "shape": (1,), "names": None},
        DONE: {"dtype": "bool", "shape": (1,), "names": None},
        PENALTY_KEY: {"dtype": "float32", "shape": (1,), "names": ["discrete_penalty"]},
        OBS_STATE: {"dtype": "float32", "shape": tuple(np.asarray(obs["agent_pos"]).shape), "names": None},
        f"{OBS_IMAGES}.front": {"dtype": "video", "shape": (3, h, w), "names": ["channels", "height", "width"]},
        f"{OBS_IMAGES}.wrist": {"dtype": "video", "shape": (3, h, w), "names": ["channels", "height", "width"]},
    }

    dataset = LeRobotDataset.create(
        args.repo_id,
        args.fps,
        root=str(root),
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
        features=features,
    )

    saved, attempts, total_steps = 0, 0, 0
    max_attempts = args.episodes * 3
    try:
        while saved < args.episodes and attempts < max_attempts:
            obs, _ = env.reset(seed=args.seed + attempts)
            expert.reset()
            attempts += 1
            success = False
            for t in range(args.max_steps):
                action = expert.act(env)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated or t == args.max_steps - 1)
                dataset.add_frame(
                    make_frame(obs, action, float(reward), done, float(info.get("discrete_penalty", 0.0)), args.task)
                )
                obs = next_obs
                if terminated or truncated:
                    success = bool(info.get("succeed", False))
                    break
            if success:
                dataset.save_episode()
                saved += 1
                total_steps += t + 1
                print(f"saved episode {saved}/{args.episodes} ({t + 1} steps)", flush=True)
            else:
                dataset.clear_episode_buffer()
                print(f"attempt {attempts}: failed, discarded", flush=True)
    finally:
        if hasattr(dataset, "finalize"):
            dataset.finalize()
        env.close()

    print(f"\nDone: {saved} episodes, {total_steps} frames, {attempts} attempts.")
    print(f"Dataset folder: {root}")
    print(f"Repo id for training: {args.repo_id}")


if __name__ == "__main__":
    main()
