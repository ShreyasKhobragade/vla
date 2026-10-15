# VLA pick-cube experiments

Imitation learning on the simulated Franka Panda pick-cube task from
[gym_hil](https://github.com/huggingface/gym-hil), comparing **ACT** and **SmolVLA**
trained with [LeRobot](https://github.com/huggingface/lerobot).

## Pipeline

1. `scripted_expert.py` – a scripted expert (hover, descend, close, lift) that reads the cube's
   position from the simulator and outputs the same `[dx, dy, dz, gripper]` actions as keyboard teleop.
2. `record_expert.py` – runs the expert and saves only the successful episodes as a LeRobotDataset
   (front + wrist cameras at 128x128, 18-D state, 4-D action, 10 fps).
3. Training uses `lerobot-train` (logs: `train_*.log`; checkpoints in `outputs/` are not in git).
4. `eval_policy.py` – evaluates a checkpoint (or the expert) on fixed seeds 1000–1049; an episode
   succeeds if the cube rises at least 5 cm. Writes CSVs, summary JSONs with Wilson 95% CIs, and GIFs to `eval_results/`.
5. `run_after_training.sh` – waits for training to finish, evaluates each checkpoint and builds a learning curve.

## Results (success at ≥5 cm, 50 seeds, dataset `local/franka_expert_v2`, 500 demos)

| Policy | Checkpoint | Success | 95% CI |
|---|---|---|---|
| Scripted expert | – | 20/20 | 84–100% |
| ACT | 25k | 47/50 (94%) | 84–98% |
| SmolVLA | 25k | 47/50 (94%) | 84–98% |
| SmolVLA | 30k | 46/50 (92%) | 81–97% |

SmolVLA learning curve: `eval_results/smolvla30k_curve.csv`.

## Setup

- LeRobot at commit `200ee535` (clone it into `lerobot/`)
- gym-hil 0.1.14

```bash
python record_expert.py --repo-id local/franka_expert_v2 --episodes 500
python eval_policy.py --expert --episodes 20 --name expert
python eval_policy.py --name smolvla --policy-type smolvla --episodes 50 \
    --checkpoint outputs/train/smolvla_v2_30k/checkpoints/030000/pretrained_model
```
