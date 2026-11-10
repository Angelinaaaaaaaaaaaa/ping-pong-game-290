#!/bin/bash
# Follow-up to launch_paper_alpha.sh, written into the same runs_paper/paper_alpha tree:
#   A. simulated IBR (paper Sec. 5.1, H_IBR = 5) against the six other decision rules, so the IBR
#      column follows the paper's definition for every Table 1 row (waits for paper_alpha/TRAIN_DONE);
#   B. self-play without choice slots, pi(x, role) as in paper Eq. 13 (runs_paper/selfplay_nochoice),
#      evaluated like the other self-play rows: Table 1 opponents incl. the paper-faithful
#      alpha-potential, simulated IBR, response gains (waits for selfplay_nochoice/TRAIN_DONE).
set -u
cd ~/angelina/ping-pong-game-290-designb
LOCK=runs_paper/paper_extra.lock
if [ -e $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then echo "already running"; exit 0; fi
echo $$ > $LOCK
PY=~/angelina/envs/designb/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
R=runs_paper
A=$R/paper_alpha
NC=$R/selfplay_nochoice
SET="--defend-mechanism preparation_cal --decision-timing contact"
echo "extra launched $(date -Is)" >> $A/MANIFEST.txt

eval_one() { P=$1; out=$2; method=$3; model=$4; workers=$5; shift 5
  mkdir -p $(dirname $out)
  [ -e $out.json ] && return 0
  $PY -u scripts/eval_table1_attack_defend.py $SET --skill-profile $P --workers $workers --method $method \
      --model-dir $model "$@" --output $out.json > $out.log 2>&1
}
gain_one() { P=$1; out=$2; method=$3; model=$4; repeats=$5; seed=$6
  mkdir -p $(dirname $out)
  [ -e $out.json ] && return 0
  $PY -u scripts/eval_response_gains.py --method $method --model-dir $model --skill-profile $P \
      --points 32 --workers 8 --repeats $repeats --seed $seed --output $out.json > $out.log 2>&1
}
export -f eval_one gain_one
export PY R A NC SET

until [ -e $A/TRAIN_DONE ]; do sleep 300; done
# ---- A. simulated IBR vs the other rules (4 chunks of 20 rallies each -> 4 workers) ------------------
{ for P in aggressive moderate_aggressive; do for s in 0 1 2; do
    M=$A/$P/phi/gamma0.99_s$s
    for rule in nash-q br-q nash-p-ibr nash-p-br nash-p-adaptive nash-p-minimax; do
      echo "eval_one $P $A/$P/ibr_sim/${rule}_s$s $rule $M 4 --seed $((70 + s)) --episodes 40 --opponents ibr-sim --split ibr-sim=20 --ibr-horizon 5"
    done
  done; done; } | xargs -P 3 -I{} bash -c '{}'
touch $A/EXTRA_A_DONE

until [ -e $NC/TRAIN_DONE ]; do sleep 300; done
# ---- B. no-choice self-play, paper format ------------------------------------------------------------
{ for P in aggressive moderate_aggressive; do for s in 0 1 2; do
    M=$A/$P/phi/gamma0.99_s$s; SP=$NC/${P}_g099_s$s.pth
    for kind in selfplay selfplay-argmax; do n=${kind/selfplay/selfplay-nc}
      echo "eval_one $P $A/$P/table1/${n}_s$s $kind:$SP $M 8 --seed $s --episodes 100 --opponents static random ibr nash-p-adaptive nash-p-argmax"
      echo "eval_one $P $A/$P/ibr_sim/${n}_s$s $kind:$SP $M 4 --seed $((70 + s)) --episodes 40 --opponents ibr-sim --split ibr-sim=20 --ibr-horizon 5"
    done
    echo "gain_one $P $A/$P/gains/selfplay-nc_s$s selfplay:$SP $M 3 $s"
    echo "gain_one $P $A/$P/gains/selfplay-nc-argmax_s$s selfplay-argmax:$SP $M 1 $s"
  done; done; } | xargs -P 4 -I{} bash -c '{}'
touch $A/EXTRA_DONE
echo "extra done $(date -Is)" >> $A/MANIFEST.txt
rm -f $LOCK
