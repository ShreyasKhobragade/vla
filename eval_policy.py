"""Evaluate a trained LeRobot ACT policy (or the scripted expert) on gym_hil Franka pick-cube.

Every episode uses a fixed seed that sets the cube position, so two policies (or one policy
under two conditions) are evaluated on exactly the same cube positions.

Success is measured here, not by the environment: an episode counts as a success if the cube
rises more than --success-lift metres (default 0.05 = 5 cm) above its starting height at any
point. The environment's own rule (more than 10 cm) is reported next to it as env_success.
The environment still ends an episode when the cube rises 10 cm or leaves the table patch.

Examples (run from the folder containing scripted_expert.py):
  # sanity check of the harness itself: the scripted expert should score about 100%
  python eval_policy.py --expert --episodes 20 --name expert

  # a trained ACT checkpoint
  python eval_policy.py --name act_smoke --episodes 50 \
      --checkpoint outputs/train/act_smoke/checkpoints/last/pretrained_model \
      --repo-id local/franka_expert_v1

  # pick-and-place (pick_place_env.py): success is the environment's own place rule
  python eval_policy.py --env pick_place --expert --episodes 20 --name pickplace_expert

Results go to eval_results/<name>.csv, <name>_summary.json and a few <name>_*.gif rollouts.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from gym_hil import make_env

ACTION_LOW = np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32)
ACTION_HIGH = np.array([1.0, 1.0, 1.0, 2.0], dtype=np.float32)
ENV_SUCCESS_LIFT = 0.10  # the environment's own success rule (metres)
LIFTED_LIFT = 0.02       # "cube was clearly picked up at all" (metres)


TASK_DEFAULTS = {  # env -> (language instruction, max steps per episode)
    "pick_cube": ("pick_cube", 100),
    "pick_place": ("pick up the cube and place it on the yellow pad", 150),
}


def make_pick_env(random_cube=True, task_env="pick_cube"):
    if task_env == "pick_place":
        from pick_place_env import make_pick_place_env

        return make_pick_place_env(image_obs=True, random_block_position=random_cube)
    return make_env(
        "gym_hil/PandaPickCubeBase-v0",
        use_viewer=False,
        use_inputs_control=False,
        reset_delay_seconds=0.0,
        gripper_penalty=-0.02,
        render_mode="rgb_array",
        image_obs=True,
        random_block_position=random_cube,
    )


class ExpertAgent:
    """The scripted expert, to check the harness itself."""

    def __init__(self, task_env="pick_cube"):
        from scripted_expert import PickCubeExpert, PickPlaceExpert

        self.expert = PickPlaceExpert() if task_env == "pick_place" else PickCubeExpert()

    def reset(self, seed=None):
        self.expert.reset()

    def act(self, env, obs):
        return self.expert.act(env)


class ACTAgent:
    """A trained LeRobot ACT checkpoint. Follows the pattern in LeRobot's ACT tutorial example."""

    def __init__(self, checkpoint, repo_id, root, device):
        import torch
        from lerobot.datasets import LeRobotDatasetMetadata
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.act import ACTPolicy

        self.torch = torch
        self.device = torch.device(device)
        self.policy = ACTPolicy.from_pretrained(checkpoint)
        self.policy.to(self.device)
        self.policy.eval()
        meta = LeRobotDatasetMetadata(repo_id, root=root)
        self.pre, self.post = make_pre_post_processors(self.policy.config, dataset_stats=meta.stats)

    def reset(self, seed=None):
        self.policy.reset()  # clears the queue of already-predicted action chunks

    def _img(self, arr):
        t = self.torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(self.device)

    def act(self, env, obs):
        torch = self.torch
        frame = {
            "observation.state": torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32)).unsqueeze(0).to(self.device),
            "observation.images.front": self._img(obs["pixels"]["front"]),
            "observation.images.wrist": self._img(obs["pixels"]["wrist"]),
        }
        batch = self.pre(frame)
        with torch.no_grad():
            action = self.policy.select_action(batch)
        action = self.post(action)
        action = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return np.clip(action, ACTION_LOW, ACTION_HIGH)


class SmolVLAAgent:
    """A fine-tuned SmolVLA checkpoint.

    Differences from ACT: the language task string is part of the input, the saved processors
    (normalization stats, tokenizer) are loaded from the checkpoint itself, and the model samples
    its action chunk from random noise, so torch is seeded per episode to keep runs repeatable.
    The camera names must match training (--rename_map front->camera1, wrist->camera2).
    """

    def __init__(self, checkpoint, task, device):
        import torch
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla import SmolVLAPolicy

        self.torch = torch
        self.task = task
        self.device = torch.device(device)
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint)
        self.policy.to(self.device)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

    def reset(self, seed=None):
        self.policy.reset()  # clears the queue of already-predicted action chunks
        if seed is not None:
            self.torch.manual_seed(seed)
            self.torch.cuda.manual_seed_all(seed)

    def _img(self, arr):
        t = self.torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(self.device)

    def act(self, env, obs):
        torch = self.torch
        frame = {
            "observation.state": torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32)).unsqueeze(0).to(self.device),
            # training renamed front -> camera1 and wrist -> camera2, so feed those names directly
            "observation.images.camera1": self._img(obs["pixels"]["front"]),
            "observation.images.camera2": self._img(obs["pixels"]["wrist"]),
            "task": self.task,
        }
        batch = self.pre(frame)
        with torch.no_grad():
            action = self.policy.select_action(batch)
        action = self.post(action)
        action = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return np.clip(action, ACTION_LOW, ACTION_HIGH)


def run_episode(env, agent, seed, max_steps, keep_frames, success_lift, pick_place=False):
    """One seeded episode. Pick-cube success: the cube rose success_lift above its start.
    Pick-and-place success: the environment's place rule fired (see pick_place_env.py)."""
    np.random.seed(seed)  # gym_hil samples the cube with the global numpy RNG
    obs, _ = env.reset(seed=seed)
    agent.reset(seed)
    u = env.unwrapped
    z_init = float(u._z_init)  # cube height at the start of the episode
    max_lift, first_step, env_success, end_reason = 0.0, None, False, "timeout"
    frames, t = [], 0
    for t in range(max_steps):
        if keep_frames:
            frames.append(np.concatenate([obs["pixels"]["front"], obs["pixels"]["wrist"]], axis=1))
        action = agent.act(env, obs)
        obs, _, terminated, truncated, info = env.step(action)
        lift = float(u._data.sensor("block_pos").data[2]) - z_init
        max_lift = max(max_lift, lift)
        if not pick_place and first_step is None and lift > success_lift:
            first_step = t + 1
        if terminated or truncated:
            env_success = bool(info.get("succeed", False))
            end_reason = "env_success" if env_success else "out_of_bounds"
            if pick_place and env_success:
                first_step = t + 1
            break
    block = u._data.sensor("block_pos").data
    row = {
        "seed": seed,
        "success": int(first_step is not None),
        "env_success": int(env_success),
        "end_reason": end_reason,
        "max_lift_cm": round(100 * max_lift, 1),
        "first_step_over_threshold": first_step if first_step is not None else "",
        "steps": t + 1,
        "final_block_z": round(float(block[2]), 4),
    }
    if pick_place:
        row["final_pad_offset_cm"] = round(100 * float(np.linalg.norm(block[:2] - u.pad_pos[:2])), 1)
    return row, frames


def wilson(k, n, z=1.96):
    """95% Wilson confidence interval for a success rate."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def rate(k, n):
    lo, hi = wilson(k, n)
    return {"count": k, "rate": round(k / n, 3), "ci95": [round(lo, 3), round(hi, 3)]}


def save_gif(frames, path):
    from PIL import Image

    ims = [Image.fromarray(f).resize((f.shape[1] * 3, f.shape[0] * 3), Image.NEAREST) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=200, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run")
    ap.add_argument("--expert", action="store_true", help="evaluate the scripted expert instead of a checkpoint")
    ap.add_argument("--checkpoint", help="path to .../checkpoints/last/pretrained_model")
    ap.add_argument("--policy-type", choices=["act", "smolvla"], default="act")
    ap.add_argument("--env", choices=list(TASK_DEFAULTS), default="pick_cube")
    ap.add_argument("--task", default=None,
                    help="language instruction (SmolVLA only; must match training; default depends on --env)")
    ap.add_argument("--repo-id", help="dataset the policy was trained on (ACT only, for normalization stats)")
    ap.add_argument("--root", default=None, help="dataset folder (default: LeRobot cache)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1000, help="first evaluation seed (training used small seeds)")
    ap.add_argument("--max-steps", type=int, default=None, help="default depends on --env")
    ap.add_argument("--success-lift", type=float, default=0.05,
                    help="metres the cube must rise to count as success (default 0.05; at most 0.10)")
    ap.add_argument("--videos", type=int, default=3, help="save GIFs of the first N episodes")
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()
    default_task, default_steps = TASK_DEFAULTS[args.env]
    args.task = args.task or default_task
    args.max_steps = args.max_steps or default_steps
    pick_place = args.env == "pick_place"

    if not 0 < args.success_lift <= ENV_SUCCESS_LIFT:
        ap.error("--success-lift must be above 0 and at most 0.10 (the environment ends episodes at 10 cm)")
    if args.expert:
        agent = ExpertAgent(args.env)
    elif args.policy_type == "smolvla":
        if not args.checkpoint:
            ap.error("give --checkpoint (SmolVLA reads its normalization stats from the checkpoint)")
        agent = SmolVLAAgent(args.checkpoint, args.task, args.device)
    else:
        if not (args.checkpoint and args.repo_id):
            ap.error("give --checkpoint and --repo-id, or use --expert")
        root = args.root or str(Path.home() / ".cache" / "huggingface" / "lerobot" / args.repo_id)
        agent = ACTAgent(args.checkpoint, args.repo_id, root, args.device)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = make_pick_env(random_cube=True, task_env=args.env)

    rows = []
    for i in range(args.episodes):
        seed = args.seed + i
        row, frames = run_episode(env, agent, seed, args.max_steps, keep_frames=i < args.videos,
                                  success_lift=args.success_lift, pick_place=pick_place)
        rows.append(row)
        if frames:
            save_gif(frames, out / f"{args.name}_seed{seed}_{'ok' if row['success'] else 'fail'}.gif")
        print(f"seed {seed}: {'success' if row['success'] else 'FAIL   '}  steps={row['steps']:3d}  "
              f"max_lift={row['max_lift_cm']:5.1f} cm  ended: {row['end_reason']}"
              + (f"  pad_offset={row['final_pad_offset_cm']:4.1f} cm" if pick_place else ""), flush=True)
    env.close()

    n = len(rows)
    k = sum(r["success"] for r in rows)
    k_env = sum(r["env_success"] for r in rows)
    k_lifted = sum(r["max_lift_cm"] > 100 * LIFTED_LIFT for r in rows)
    k_oob = sum(r["end_reason"] == "out_of_bounds" for r in rows)
    ok_steps = [r["first_step_over_threshold"] for r in rows if r["success"]]
    if pick_place:
        head = {"env": args.env, "task": args.task, "success_placed": rate(k, n)}
    else:
        head = {
            "success_threshold_cm": round(100 * args.success_lift, 1),
            f"success_over_{round(100 * args.success_lift, 1)}cm": rate(k, n),
            "env_success_over_10cm": rate(k_env, n),
        }
    summary = {
        "name": args.name,
        "episodes": n,
        **head,
        f"lifted_over_{round(100 * LIFTED_LIFT, 1)}cm": rate(k_lifted, n),
        "ended_out_of_bounds": k_oob,
        "mean_step_when_threshold_reached": round(float(np.mean(ok_steps)), 1) if ok_steps else None,
        "first_seed": args.seed,
        "max_steps": args.max_steps,
    }
    with open(out / f"{args.name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out / f"{args.name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lo, hi = wilson(k, n)
    if pick_place:
        print(f"\n{args.name}: {k}/{n} placed on the pad = {100 * k / n:.0f}%  "
              f"(95% CI {100 * lo:.0f}% to {100 * hi:.0f}%)")
    else:
        print(f"\n{args.name}: {k}/{n} reached {100 * args.success_lift:.0f} cm = {100 * k / n:.0f}%  "
              f"(95% CI {100 * lo:.0f}% to {100 * hi:.0f}%)")
        print(f"  environment's own 10 cm rule: {k_env}/{n}")
    print(f"  cube lifted at all (over {100 * LIFTED_LIFT:.0f} cm): {k_lifted}/{n}")
    print(f"  episodes ended by cube leaving the table patch: {k_oob}/{n}")
    print(f"results in {out}/")


if __name__ == "__main__":
    main()
