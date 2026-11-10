#!/bin/bash
# Paper-faithful Design B runs, both skill profiles (2026-10-25).
# Setting for everything below: retrained controller (logs/best_model_tracker1 on the
# server, md5 db5df579...), calibrated preparation poses (preparation_cal), decisions at
# racket contacts (decision_timing contact), target reset, 2500-step cap.
#   0  checks: does calibrated preparation matter (one-sided), rally length (both-sided)
#   1  data: per profile 12 grid + 12 mixed shards x 1500 decisions
#   S  self-play, per-player + role-gated: 3 seeds gamma 0.99, + gamma 0.9 / 0.7 seed 0
#   2  Phi/Q: gamma {0.99 x3 seeds, 0.95, 0.9, 0.7}, data volume, selection and loss ablations
#   3  Table 1: static + random + Q-IBR (+ simulated IBR for main methods), head-to-heads
set -u
cd ~/angelina/ping-pong-game-290-designb
PY=~/angelina/envs/designb/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
R=runs_paper
PROFILES="aggressive moderate_aggressive"
SET="--defend-mechanism preparation_cal --decision-timing contact"
mkdir -p $R
{ echo "launched: $(date -Is)"; md5sum logs/best_model_tracker1/best_model.zip; } > $R/MANIFEST.txt

# ---- 0: mechanism checks (background) --------------------------------------------
mkdir -p $R/checks
for P in $PROFILES; do
  M=$([ $P = aggressive ] && echo "preparation preparation_cal" || echo "preparation_cal")
  ( $PY -u scripts/compare_defend_mechanisms.py --mode one-sided --pairs 30 --workers 6 --skill-profile $P \
        --mechanisms $M --output $R/checks/one_sided_$P.json > $R/checks/one_sided_$P.log 2>&1
    $PY -u scripts/compare_defend_mechanisms.py --mode both-sided --pairs 200 --workers 6 --skill-profile $P \
        --mechanisms $M --output $R/checks/both_sided_$P.json > $R/checks/both_sided_$P.log 2>&1 ) &
done

# ---- S: self-play (background, longest) ------------------------------------------
mkdir -p $R/selfplay
sp() { P=$1; name=$2; shift 2
  $PY -u nash_skills/v2/selfplay_per_player.py --iterations 1000 --rallies-per-iter 32 --round-iters 25 \
      --defend-mechanism preparation_cal --skill-profile $P --reset-targets "$@" \
      --output $R/selfplay/${P}_$name.pth --log $R/selfplay/${P}_$name.csv > $R/selfplay/${P}_$name.log 2>&1 & }
for P in $PROFILES; do
  for s in 0 1 2; do sp $P g099_s$s --gamma 0.99 --seed $s; done
  sp $P g09_s0 --gamma 0.9 --seed 0
  sp $P g07_s0 --gamma 0.7 --seed 0
done

# ---- 1: data -----------------------------------------------------------------------
pids=()
for P in $PROFILES; do
  mkdir -p $R/$P/data_shards
  for i in $(seq 0 11); do
    for mode in grid mixed; do
      base=$([ $mode = grid ] && echo 700 || echo 800)
      $PY -u nash_skills/v2/collect_data_attack_defend.py --sampling $mode --decisions 1500 --max-rallies 3000 \
          --max-steps 2500 $SET --skill-profile $P --reset-targets --seed $((base + i)) --progress-every 0 \
          --output $R/$P/data_shards/${mode}_$i.pkl > $R/$P/data_shards/${mode}_$i.log 2>&1 &
      pids+=($!)
    done
  done
done
wait "${pids[@]}"
for P in $PROFILES; do
  $PY scripts/merge_rally_shards.py --output $R/$P/rallies.pkl $R/$P/data_shards/*.pkl > $R/$P/merge.log 2>&1
done
touch $R/DATA_DONE

# ---- 2: Phi/Q ------------------------------------------------------------------------
pids=()
phi() { P=$1; name=$2; shift 2
  mkdir -p $R/$P/phi
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -u nash_skills/v2/train_q_model_5skill_v3_attack_defend.py \
      --rallies $R/$P/rallies.pkl --theta-mode mixed "$@" --out-dir $R/$P/phi/$name > $R/$P/phi/$name.log 2>&1 &
  pids+=($!); }
for P in $PROFILES; do
  NTR=$($PY -c "import pickle; n=len(pickle.load(open('$R/$P/rallies.pkl','rb'))); print(int(0.8*0.9*n))")
  V="--phi-loss lse --phi-select val_residual"
  for s in 0 1 2; do phi $P g099_s$s $V --gamma 0.99 --seed $s; done
  for g in 0.95 0.9 0.7; do phi $P g${g/./}_s0 $V --gamma $g --seed 0; done
  for f in 8 4 2; do phi $P g099_n$((NTR / f))_s0 $V --gamma 0.99 --seed 0 --train-rally-limit $((NTR / f)); done
  phi $P g099_trainloss_s0 --phi-loss lse --phi-select train_loss --gamma 0.99 --seed 0
  phi $P g099_mse_s0 --phi-loss mse --phi-select val_residual --gamma 0.99 --seed 0
done
wait "${pids[@]}"
touch $R/PHI_DONE

# ---- 3a: Table 1 for every Phi run (static + random + Q-IBR) ---------------------
E="$PY -u scripts/eval_table1_attack_defend.py $SET"
t1() { P=$1; name=$2; seed=$3; shift 3
  mkdir -p $R/$P/table1
  $E --skill-profile $P --workers 12 --method nash-p-pga --model-dir $R/$P/phi/$name --seed $seed "$@" \
     --output $R/$P/table1/pga_$name.json > $R/$P/table1/pga_$name.log 2>&1; }
pids=()
for P in $PROFILES; do
  ( for s in 0 1 2; do t1 $P g099_s$s $s --episodes 100 --opponents static random ibr; done ) &
  pids+=($!)
  ( for d in $R/$P/phi/*/; do n=$(basename $d); case $n in g099_s?) ;; *) t1 $P $n 0 --episodes 100 --opponents static random ibr;; esac; done ) &
  pids+=($!)
done
wait "${pids[@]}"
touch $R/TABLE1_PHI_DONE
echo "phase 3a done: $(date -Is)" >> $R/MANIFEST.txt
