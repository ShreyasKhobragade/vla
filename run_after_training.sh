#!/usr/bin/env bash
# Waits for the SmolVLA 30k training run to end, then evaluates each saved checkpoint
# on the same 50 seeded cubes and prints a results table. Progress goes to logs/eval_after_training.log.
cd ~/vla || exit 1
RUN=outputs/train/smolvla_v2_30k
exec >> logs/eval_after_training.log 2>&1

echo "[$(date)] waiting for training to finish"
while pgrep -f "bin/lerobot-train" > /dev/null; do sleep 60; done
echo "[$(date)] training process ended"

if [ -d "$RUN/checkpoints/030000" ]; then
  echo "training reached 30000 steps"
else
  echo "WARNING: no 030000 checkpoint, training did not finish. Evaluating the checkpoints that exist."
fi
sleep 30  # let the GPU release its memory

for s in 005000 010000 015000 020000 025000 030000; do
  ckpt="$RUN/checkpoints/$s/pretrained_model"
  if [ ! -d "$ckpt" ]; then echo "skip $s (not found)"; continue; fi
  videos=0; [ "$s" = "030000" ] && videos=3
  echo "[$(date)] evaluating $s"
  python eval_policy.py --name smolvla30k_$s --policy-type smolvla --episodes 50 \
    --videos $videos --checkpoint "$ckpt" || echo "FAILED: $s"
done

python - <<'PY'
import glob, json, csv
rows = []
for f in sorted(glob.glob("eval_results/smolvla30k_*_summary.json")):
    d = json.load(open(f))
    key = next(k for k in d if k.startswith("success_over_"))
    r = d[key]
    step = d["name"].split("_")[-1]
    rows.append((int(step), r["count"], d["episodes"], r["rate"], r["ci95"]))
print("\nSmolVLA 30k run, success over 5 cm on seeds 1000-1049")
print("step     success   rate   95% CI")
for step, k, n, rate, ci in rows:
    print(f"{step:>6}   {k:>2}/{n}   {100*rate:4.0f}%   {100*ci[0]:.0f}% to {100*ci[1]:.0f}%")
with open("eval_results/smolvla30k_curve.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["step", "successes", "episodes", "rate", "ci_lo", "ci_hi"])
    for step, k, n, rate, ci in rows: w.writerow([step, k, n, rate, ci[0], ci[1]])
PY
echo "[$(date)] all done"
