#!/bin/bash
# Phase 4 of the paper-faithful runs (after scripts/launch_paper_runs.sh):
#   self-play Table 1 rows (sampled + argmax, 3 seeds; gamma ablation seeds),
#   simulated IBR column for every main method, alpha-potential vs self-play head-to-heads.
set -u
cd ~/angelina/ping-pong-game-290-designb
PY=~/angelina/envs/designb/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
R=runs_paper
PROFILES="aggressive moderate_aggressive"
SET="--defend-mechanism preparation_cal --decision-timing contact"
E="$PY -u scripts/eval_table1_attack_defend.py $SET"

until [ -e $R/PHI_DONE ] && [ $(grep -l "^Done\." $R/selfplay/*.log 2>/dev/null | wc -l) -ge 10 ]; do sleep 120; done
echo "phase 4 start: $(date -Is)" >> $R/MANIFEST.txt

run_profile() { P=$1; T=$R/$P/table1; mkdir -p $T $R/$P/ibr_sim $R/$P/h2h
  ev() { out=$1; shift; [ -e $out.json ] && return 0
         $E --skill-profile $P --workers 24 "$@" --output $out.json > $out.log 2>&1; }
  for s in 0 1 2; do
    for kind in selfplay selfplay-argmax; do
      ev $T/${kind}_g099_s$s --method $kind:$R/selfplay/${P}_g099_s$s.pth --model-dir $R/$P/phi/g099_s$s \
         --seed $s --episodes 100 --opponents static random ibr
    done
  done
  for g in g09_s0 g07_s0; do
    ev $T/selfplay_$g --method selfplay:$R/selfplay/${P}_$g.pth --seed 0 --episodes 100 --opponents static random
  done
  for s in 0 1 2; do
    ev $R/$P/h2h/pga_vs_selfplay_s$s --method nash-p-pga --model-dir $R/$P/phi/g099_s$s --seed $((60 + s)) \
       --episodes 100 --opponents selfplay:$R/selfplay/${P}_g099_s$s.pth selfplay-argmax:$R/selfplay/${P}_g099_s$s.pth
  done
  for s in 0 1 2; do
    I="--episodes 40 --opponents ibr-sim --split ibr-sim=20 --seed $((70 + s))"
    ev $R/$P/ibr_sim/pga_g099_s$s --method nash-p-pga --model-dir $R/$P/phi/g099_s$s $I
    ev $R/$P/ibr_sim/selfplay_g099_s$s --method selfplay:$R/selfplay/${P}_g099_s$s.pth $I
    ev $R/$P/ibr_sim/selfplay-argmax_g099_s$s --method selfplay-argmax:$R/selfplay/${P}_g099_s$s.pth $I
  done
}
for P in $PROFILES; do run_profile $P & done
wait
touch $R/ALL_DONE
echo "all done: $(date -Is)" >> $R/MANIFEST.txt
