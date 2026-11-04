#!/bin/bash
# Simulated-IBR column for the alpha-potential rows (does not need self-play); started
# early while self-play trains. launch_paper_eval.sh skips outputs that already exist.
set -u
cd ~/angelina/ping-pong-game-290-designb
PY=~/angelina/envs/designb/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
R=runs_paper
E="$PY -u scripts/eval_table1_attack_defend.py --defend-mechanism preparation_cal --decision-timing contact"
for P in aggressive moderate_aggressive; do
  ( mkdir -p $R/$P/ibr_sim
    for s in 0 1 2; do
      out=$R/$P/ibr_sim/pga_g099_s$s
      [ -e $out.json ] || $E --skill-profile $P --workers 20 --method nash-p-pga --model-dir $R/$P/phi/g099_s$s \
          --episodes 40 --opponents ibr-sim --split ibr-sim=20 --seed $((70 + s)) --output $out.json > $out.log 2>&1
    done ) &
done
wait
