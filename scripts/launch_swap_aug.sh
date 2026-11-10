#!/bin/bash
# Player-swap augmentation for alpha-potential (base data, preparation_cal): does a side-symmetric
# Phi/Q raise the static minimum? gamma 0.99, Q early stopping, 3 seeds, both profiles; PGA and BR-Q
# vs static / random / Q-IBR / self-play (sampled), 100 decided rallies per side.
set -u
cd ~/angelina/ping-pong-game-290-designb
LOCK=runs_paper/swap_aug.lock
if [ -e $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then echo "already running"; exit 0; fi
echo $$ > $LOCK
PY=~/angelina/envs/designb/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
R=runs_paper
W=$R/swap_aug
mkdir -p $W
pids=()
for P in aggressive moderate_aggressive; do
  for s in 0 1 2; do
    mkdir -p $W/$P/phi
    OMP_NUM_THREADS=2 $PY -u nash_skills/v2/train_q_model_5skill_v3_attack_defend.py --rallies $R/$P/rallies.pkl \
        --theta-mode mixed --phi-loss lse --phi-select val_residual --q-select val --augment-swap \
        --gamma 0.99 --seed $s --out-dir $W/$P/phi/g099_s$s > $W/$P/phi/g099_s$s.log 2>&1 &
    pids+=($!)
  done
done
wait "${pids[@]}"
run_one() { P=$1; rule=$2; s=$3
  out=$W/$P/table1/${rule}_s$s; mkdir -p $(dirname $out)
  [ -e $out.json ] && return 0
  $PY -u scripts/eval_table1_attack_defend.py --defend-mechanism preparation_cal --decision-timing contact \
      --skill-profile $P --workers 10 --method $rule --model-dir $W/$P/phi/g099_s$s --seed $s --episodes 100 \
      --opponents static random ibr selfplay:$R/selfplay/${P}_g099_s$s.pth --output $out.json > $out.log 2>&1
}
export -f run_one
export PY W R
for P in aggressive moderate_aggressive; do for rule in nash-p-pga br-q; do for s in 0 1 2; do echo "$P $rule $s"; done; done; done \
  | xargs -P 4 -L 1 bash -c 'run_one "$@"' _
touch $W/DONE
rm -f $LOCK
