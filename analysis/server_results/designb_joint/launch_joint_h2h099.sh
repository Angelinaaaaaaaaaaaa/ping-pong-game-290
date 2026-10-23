#!/bin/bash
# Head-to-head: self-play (sampled / argmax, 3 seeds) vs alpha-potential PGA with gamma 0.99 (Phi seed 0).
set -u
cd ~/angelina/ping-pong-game-290-designb
PY=~/angelina/envs/designb/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
R=runs_joint
until [ -e $R/ALL_DONE ]; do sleep 60; done
E="$PY -u scripts/eval_table1_attack_defend.py --episodes 100 --workers 2 --defend-mechanism preparation --model-dir $R/phi/gamma099_s0 --opponents nash-p-pga"
pids=()
for kind in selfplay selfplay-argmax; do
  tag=$([ $kind = selfplay ] && echo sampled || echo argmax)
  for s in 0 1 2; do
    $E --method $kind:$R/selfplay/prep_seed$s.pth --seed $((40 + s)) \
       --output $R/table1_gamma099/h2h_selfplay_${tag}_s$s.json > $R/table1_gamma099/h2h_selfplay_${tag}_s$s.log 2>&1 &
    pids+=($!)
  done
done
wait "${pids[@]}"
touch $R/H2H099_DONE
