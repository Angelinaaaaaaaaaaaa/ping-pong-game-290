#!/bin/bash
# alpha-potential exactly as in the paper's Sec. 4.1, plus everything Sec. 5 reports, both profiles:
#   policy family: one fixed (target, preparation) pair per player (Eq. 7) -> fixed-pair rallies only,
#   value targets with continuation length H = 10 exchanges (Eq. 9), smooth-max potential loss (Eq. 11),
#   online phase: best of the 225 joint fixed-pair profiles by Phi (Eq. 12, rule nash-p-argmax).
# Phases: Phi/Q training (gamma sweep + data volume) -> figure evals -> Table 1 -> Table 2 -> simulated IBR.
set -u
cd ~/angelina/ping-pong-game-290-designb
LOCK=runs_paper/paper_alpha.lock
if [ -e $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then echo "already running"; exit 0; fi
echo $$ > $LOCK
PY=~/angelina/envs/designb/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
R=runs_paper
A=$R/paper_alpha
mkdir -p $A
echo "launched $(date -Is)" >> $A/MANIFEST.txt
PHI="--theta-mode pure --pure-data-only --horizon 10 --phi-loss lse --phi-select val_residual --q-select val"
SET="--defend-mechanism preparation_cal --decision-timing contact"
GAMMAS="0.5 0.6 0.7 0.8 0.9 0.95 0.99 1.0"
VOLUMES="170 550 1700 5500"
RULES="nash-q br-q nash-p-ibr nash-p-br nash-p-adaptive nash-p-minimax nash-p-argmax"

train_one() { P=$1; name=$2; shift 2
  d=$A/$P/phi/$name; mkdir -p $A/$P/phi
  [ -e $d/heldout_attack_defend.json ] && return 0
  OMP_NUM_THREADS=2 $PY -u nash_skills/v2/train_q_model_5skill_v3_attack_defend.py --rallies $R/$P/rallies_10x.pkl \
      $PHI "$@" --out-dir $d > $d.log 2>&1
}
eval_one() { P=$1; out=$2; method=$3; model=$4; shift 4
  mkdir -p $(dirname $out)
  [ -e $out.json ] && return 0
  $PY -u scripts/eval_table1_attack_defend.py $SET --skill-profile $P --workers 8 --method $method \
      --model-dir $model "$@" --output $out.json > $out.log 2>&1
}
gain_one() { P=$1; out=$2; method=$3; model=$4; repeats=$5; seed=$6
  mkdir -p $(dirname $out)
  [ -e $out.json ] && return 0
  $PY -u scripts/eval_response_gains.py --method $method --model-dir $model --skill-profile $P \
      --points 32 --workers 8 --repeats $repeats --seed $seed --output $out.json > $out.log 2>&1
}
export -f train_one eval_one gain_one
export PY R A PHI SET

# ---- 1. Phi / Q training ------------------------------------------------------------------
{ for P in aggressive moderate_aggressive; do for s in 0 1 2; do
    for g in $GAMMAS; do echo "train_one $P gamma${g}_s$s --gamma $g --seed $s"; done
    for n in $VOLUMES; do echo "train_one $P vol${n}_s$s --gamma 0.99 --seed $s --train-rally-limit $n"; done
  done; done; } | xargs -P 16 -I{} bash -c '{}'
# the full-data point of the volume curve is the main gamma 0.99 run
for P in aggressive moderate_aggressive; do for s in 0 1 2; do
  N=$($PY -c "import json; print(json.load(open('$A/$P/phi/gamma0.99_s$s/heldout_attack_defend.json'))['n_train_rallies'])")
  ( cd $A/$P/phi && ln -sfn gamma0.99_s$s vol${N}_s$s )
done; done
touch $A/TRAIN_DONE

# ---- 2-5. evaluations, one queue --------------------------------------------------------------
{ for P in aggressive moderate_aggressive; do for s in 0 1 2; do
    M=$A/$P/phi/gamma0.99_s$s; SP=$R/selfplay/${P}_g099_s$s.pth
    for d in $A/$P/phi/gamma*_s$s $A/$P/phi/vol*_s$s; do n=$(basename $d)
      echo "eval_one $P $A/$P/figevals/$n nash-p-argmax $d --seed $s --episodes 60 --opponents static random"; done
    for rule in $RULES; do
      echo "eval_one $P $A/$P/table1/${rule}_s$s $rule $M --seed $s --episodes 100 --opponents static random ibr nash-p-adaptive selfplay:$SP"; done
    for kind in selfplay selfplay-argmax; do
      echo "eval_one $P $A/$P/table1/${kind}_s$s $kind:$SP $M --seed $s --episodes 100 --opponents static random ibr nash-p-adaptive nash-p-argmax"; done
  done; done
  for P in aggressive moderate_aggressive; do for s in 0 1 2; do
    M=$A/$P/phi/gamma0.99_s$s; SP=$R/selfplay/${P}_g099_s$s.pth
    for spec in nash-p-argmax:1 nash-q:3 br-q:2 nash-p-ibr:1 nash-p-br:1 nash-p-adaptive:1 nash-p-minimax:1; do
      rule=${spec%%:*}; rep=${spec##*:}
      echo "gain_one $P $A/$P/gains/${rule}_s$s $rule $M $rep $s"; done
    echo "gain_one $P $A/$P/gains/selfplay_s$s selfplay:$SP $M 3 $s"
    echo "gain_one $P $A/$P/gains/selfplay-argmax_s$s selfplay-argmax:$SP $M 1 $s"
  done; done
  for P in aggressive moderate_aggressive; do for s in 0 1 2; do
    M=$A/$P/phi/gamma0.99_s$s; SP=$R/selfplay/${P}_g099_s$s.pth
    for m in nash-p-argmax selfplay:$SP selfplay-argmax:$SP; do n=$(echo ${m%%:*});
      echo "eval_one $P $A/$P/ibr_sim/${n}_s$s $m $M --seed $((70 + s)) --episodes 40 --opponents ibr-sim --split ibr-sim=20 --ibr-horizon 5"; done
  done; done; } | xargs -P 5 -I{} bash -c '{}'
touch $A/ALL_DONE
echo "done $(date -Is)" >> $A/MANIFEST.txt
rm -f $LOCK
