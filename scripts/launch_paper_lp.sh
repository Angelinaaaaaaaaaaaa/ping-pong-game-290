#!/bin/bash
# Kexin's V3 way of fitting Phi -- paper Eq. 11 solved exactly per state as a linear program, then a network
# regressed on those per-state potentials (scripts/fit_potential_lp.py) -- on the Design B joint game, reusing the
# data and the Q networks of runs_v3/paper (gamma 0.99, 3 seeds). Evaluated like Table 1 for the five
# potential-based rules (Eq. 12 argmax and Kexin's four), with the head-to-head against self-play, response
# gains and simulated IBR for Eq. 12 and Potential Adaptive Match, and the stall cell played to the end.
set -u
cd ~/angelina/ping-pong-game-290-designb-v3
LOCK=runs_v3/paper_lp.lock
if [ -e $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then echo "already running"; exit 0; fi
echo $$ > $LOCK
PY=~/angelina/envs/designb/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
P=aggressive
R=runs_v3
SRC=$R/paper
A=$R/paper_lp
SET="--defend-mechanism preparation_cal --decision-timing contact"
CAP="--max-attempt-factor 3"
RULES="nash-p-argmax nash-p-adaptive nash-p-ibr nash-p-br nash-p-minimax"
mkdir -p $A/$P/phi
echo "launched $(date -Is)" >> $A/MANIFEST.txt

for s in 0 1 2; do
  d=$A/$P/phi/gamma0.99_s$s
  [ -e $d/heldout_attack_defend.json ] && continue
  $PY -u scripts/fit_potential_lp.py --rallies $R/data/rallies.pkl --model-dir $SRC/$P/phi/gamma0.99_s$s \
      --seed $s --out-dir $d --workers 8 > $d.log 2>&1 &
done
wait
for s in 0 1 2; do
  if [ ! -e $A/$P/phi/gamma0.99_s$s/heldout_attack_defend.json ]; then
    echo "fit failed for seed $s, see $A/$P/phi/gamma0.99_s$s.log" >> $A/MANIFEST.txt; rm -f $LOCK; exit 1
  fi
done
touch $A/FIT_DONE

eval_one() { out=$1; method=$2; model=$3; shift 3
  mkdir -p $(dirname $out)
  [ -e $out.json ] && return 0
  $PY -u scripts/eval_table1_attack_defend.py $SET --skill-profile $P --workers 8 --method $method \
      --model-dir $model $CAP "$@" --output $out.json > $out.log 2>&1
}
gain_one() { out=$1; method=$2; model=$3; seed=$4
  mkdir -p $(dirname $out)
  [ -e $out.json ] && return 0
  $PY -u scripts/eval_response_gains.py --method $method --model-dir $model --skill-profile $P \
      --points 32 --workers 8 --repeats 1 --seed $seed --output $out.json > $out.log 2>&1
}
replay_one() { out=$1; method=$2; model=$3; seed=$4
  mkdir -p $(dirname $out)
  [ -e $out.json ] && return 0
  $PY -u scripts/diagnose_unfinished_rallies.py --method $method --model-dir $model \
      --opponents center_safe/center --rallies-per-side 30 --max-steps 20000 --workers 4 --seed $((80 + seed)) \
      --output $out.json > $out.log 2>&1
}
export -f eval_one gain_one replay_one
export PY P R A SET CAP

{ for s in 0 1 2; do M=$A/$P/phi/gamma0.99_s$s; SP=$R/selfplay/${P}_s$s.pth
    for rule in $RULES; do
      echo "eval_one $A/$P/table1/${rule}_s$s $rule $M --seed $s --episodes 100 --opponents static random ibr nash-p-adaptive selfplay:$SP"
    done
  done
  for s in 0 1 2; do M=$A/$P/phi/gamma0.99_s$s
    for rule in $RULES; do echo "replay_one $A/$P/diag_unfinished/${rule}_s$s $rule $M $s"; done
    for rule in nash-p-argmax nash-p-adaptive; do echo "gain_one $A/$P/gains/${rule}_s$s $rule $M $s"; done
  done
  for s in 0 1 2; do M=$A/$P/phi/gamma0.99_s$s
    for rule in nash-p-argmax nash-p-adaptive; do
      echo "eval_one $A/$P/ibr_sim/${rule}_s$s $rule $M --seed $((70 + s)) --episodes 40 --opponents ibr-sim --split ibr-sim=20 --ibr-horizon 5"
    done
  done; } | xargs -P 5 -I{} bash -c '{}'
touch $A/ALL_DONE
echo "done $(date -Is)" >> $A/MANIFEST.txt
rm -f $LOCK
