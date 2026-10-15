#!/usr/bin/env bash
# Overnight pick-and-place pipeline:
#   wait for demo recording -> train ACT -> evaluate ACT -> train SmolVLA -> evaluate SmolVLA -> summary.
# Same hyperparameters as the pick-cube runs (batch 16, chunks of 10, 30k steps, a checkpoint every 5k).
# Progress goes to logs/pickplace_overnight.log; each training run has its own log in logs/.
#
#   systemd-inhibit --what=sleep:idle:handle-lid-switch --who=training --why="pick-place training" \
#     setsid nohup ./run_pickplace_overnight.sh > /dev/null 2>&1 &
cd ~/vla || exit 1
exec >> logs/pickplace_overnight.log 2>&1
set -o pipefail

BIN=~/miniconda3/envs/vla/bin
REPO=local/franka_pickplace_v1
ROOT=$HOME/.cache/huggingface/lerobot/$REPO
EPISODES=500
ACT_RUN=outputs/train/act_pickplace_30k
SMOL_RUN=outputs/train/smolvla_pickplace_30k
STEPS="005000 010000 015000 020000 025000 030000"

log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for demo recording to finish"
while pgrep -f "record_expert.py --env pick_place" > /dev/null; do sleep 30; done
saved=$("$BIN/python" -c "import json; print(json.load(open('$ROOT/meta/info.json'))['total_episodes'])" 2>/dev/null)
if [ "$saved" != "$EPISODES" ]; then
  log "ABORT: dataset has '$saved' episodes, expected $EPISODES. See logs/record_pickplace_v1.log"
  exit 1
fi
log "dataset ok: $saved episodes"

# ---------------------------------------------------------------- ACT
log "training ACT -> $ACT_RUN"
"$BIN/lerobot-train" \
  --dataset.repo_id=$REPO --dataset.root=$ROOT \
  --policy.type=act --policy.chunk_size=10 --policy.n_action_steps=10 \
  --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=16 --steps=30000 --save_freq=5000 \
  --output_dir=$ACT_RUN --job_name=act_pickplace_30k --wandb.enable=false \
  > logs/train_act_pickplace_30k.log 2>&1 \
  && log "ACT training finished" || log "ACT training FAILED (exit $?), evaluating whatever checkpoints exist"
sleep 20

for s in $STEPS; do
  ckpt="$ACT_RUN/checkpoints/$s/pretrained_model"
  [ -d "$ckpt" ] || { log "skip ACT $s (not found)"; continue; }
  videos=0; [ "$s" = "030000" ] && videos=3
  log "evaluating ACT $s"
  "$BIN/python" eval_policy.py --env pick_place --name pp_act_$s --episodes 50 --videos $videos \
    --checkpoint "$ckpt" --repo-id $REPO 2>&1 | grep -E "^(seed|pp_act|  )" || log "ACT eval FAILED: $s"
done

# ---------------------------------------------------------------- SmolVLA
log "training SmolVLA -> $SMOL_RUN"
"$BIN/lerobot-train" \
  --policy.path=lerobot/smolvla_base --policy.push_to_hub=false --policy.device=cuda \
  --policy.chunk_size=10 --policy.n_action_steps=10 --policy.empty_cameras=1 \
  --dataset.repo_id=$REPO --dataset.root=$ROOT \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size=16 --steps=30000 --save_freq=5000 --log_freq=100 \
  --output_dir=$SMOL_RUN --job_name=smolvla_pickplace_30k --wandb.enable=false \
  > logs/train_smolvla_pickplace_30k.log 2>&1
status=$?
# resume from the last checkpoint if the run died before 30k (up to 3 times)
for attempt in 1 2 3; do
  [ -d "$SMOL_RUN/checkpoints/030000" ] && break
  if [ ! -e "$SMOL_RUN/checkpoints/last" ]; then
    log "SmolVLA training FAILED (exit $status) with no checkpoint to resume from"
    break
  fi
  log "SmolVLA stopped (exit $status) at $(readlink $SMOL_RUN/checkpoints/last); resume attempt $attempt"
  sleep 30
  "$BIN/lerobot-train" --config_path=$SMOL_RUN/checkpoints/last/pretrained_model/train_config.json --resume=true \
    > logs/train_smolvla_pickplace_30k_resume$attempt.log 2>&1
  status=$?
done
[ -d "$SMOL_RUN/checkpoints/030000" ] && log "SmolVLA training finished"
sleep 20

for s in $STEPS; do
  ckpt="$SMOL_RUN/checkpoints/$s/pretrained_model"
  [ -d "$ckpt" ] || { log "skip SmolVLA $s (not found)"; continue; }
  videos=0; [ "$s" = "030000" ] && videos=3
  log "evaluating SmolVLA $s"
  "$BIN/python" eval_policy.py --env pick_place --name pp_smolvla_$s --policy-type smolvla --episodes 50 \
    --videos $videos --checkpoint "$ckpt" 2>&1 | grep -E "^(seed|pp_smolvla|  )" || log "SmolVLA eval FAILED: $s"
done

# ---------------------------------------------------------------- summary
"$BIN/python" - <<'PY'
import csv, glob, json
rows = []
for f in sorted(glob.glob("eval_results/pp_*_summary.json")):
    d = json.load(open(f))
    policy, step = d["name"].split("_")[1], int(d["name"].split("_")[2])
    r = d["success_placed"]
    rows.append((policy, step, r["count"], d["episodes"], r["rate"], *r["ci95"]))
print("\nPick-and-place, cube placed on the pad, seeds 1000-1049")
print("policy    step    success   rate   95% CI")
for p, s, k, n, rate, lo, hi in rows:
    print(f"{p:8s} {s:>6}   {k:>2}/{n}    {100*rate:4.0f}%   {100*lo:.0f}% to {100*hi:.0f}%")
with open("eval_results/pickplace_curve.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["policy", "step", "successes", "episodes", "rate", "ci_lo", "ci_hi"])
    w.writerows(rows)
PY
log "all done"
