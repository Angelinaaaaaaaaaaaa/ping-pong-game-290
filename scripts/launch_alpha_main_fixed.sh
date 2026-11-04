#!/bin/bash
# Table 1 alpha-potential rows with the fixed Q (early stopping), 3 seeds, both profiles:
# full opponents (static + random + Q-IBR), simulated IBR, head-to-head vs the self-play runs.
# Reuses the gamma 0.99 models of the ablation sweep (runs_paper/ablations_qval/<P>/phi/gamma0.99_s<s>).
set -u
cd ~/angelina/ping-pong-game-290-designb
LOCK=runs_paper/alpha_main_fixed.lock
if [ -e $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then echo "already running"; exit 0; fi
echo $$ > $LOCK
PY=~/angelina/envs/designb/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
R=runs_paper
A=$R/ablations_qval
M=$R/alpha_fixed
E="$PY -u scripts/eval_table1_attack_defend.py --defend-mechanism preparation_cal --decision-timing contact"
until [ -e $A/GAMMA_DONE ]; do sleep 120; done
mkdir -p $M; echo "start $(date -Is)" >> $M/MANIFEST.txt
run_profile() { P=$1; mkdir -p $M/$P
  ev() { out=$1; shift; [ -e $out.json ] && return 0
         $E --skill-profile $P --workers 16 "$@" --output $out.json > $out.log 2>&1; }
  for s in 0 1 2; do
    D=$A/$P/phi/gamma0.99_s$s
    ev $M/$P/table1_s$s --method nash-p-pga --model-dir $D --seed $s --episodes 100 --opponents static random ibr
    ev $M/$P/h2h_s$s --method nash-p-pga --model-dir $D --seed $((60 + s)) --episodes 100 \
       --opponents selfplay:$R/selfplay/${P}_g099_s$s.pth selfplay-argmax:$R/selfplay/${P}_g099_s$s.pth
  done
  for s in 0 1 2; do
    ev $M/$P/ibr_sim_s$s --method nash-p-pga --model-dir $A/$P/phi/gamma0.99_s$s \
       --episodes 40 --opponents ibr-sim --split ibr-sim=20 --seed $((70 + s))
  done
}
for P in aggressive moderate_aggressive; do run_profile $P & done
wait
touch $M/ALL_DONE
rm -f $LOCK
