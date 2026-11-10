#!/bin/bash
# alpha-potential with randomized policy parameters (theta^i = a distribution over the 5 targets and
# one over the 3 preparations, instead of paper Eq. 7's single fixed pair), otherwise the protocol of
# launch_paper_alpha.sh: H = 10, smooth-max Phi loss, validation-selected checkpoints, gamma 0.99.
# Online phase: projected gradient ascent on Phi over the mixed profiles (nash-p-pga), each player
# sampling its pair from its own distribution. Question: does a mixed profile remove Eq. 12's
# exploitability (diagnose_potential_equilibrium.py: the learned game's equilibrium is mixed in most states)?
set -u
cd ~/angelina/ping-pong-game-290-designb
LOCK=runs_paper/paper_alpha_mixed.lock
if [ -e $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then echo "already running"; exit 0; fi
echo $$ > $LOCK
PY=~/angelina/envs/designb/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
R=runs_paper
A=$R/paper_alpha_mixed
NC=$R/selfplay_nochoice
mkdir -p $A
echo "launched $(date -Is)" >> $A/MANIFEST.txt
PHI="--theta-mode mixed --horizon 10 --phi-loss lse --phi-select val_residual --q-select val --gamma 0.99"
SET="--defend-mechanism preparation_cal --decision-timing contact"

for P in aggressive moderate_aggressive; do for s in 0 1 2; do
  d=$A/$P/phi/gamma0.99_s$s; mkdir -p $A/$P/phi
  [ -e $d/heldout_attack_defend.json ] && continue
  OMP_NUM_THREADS=2 $PY -u nash_skills/v2/train_q_model_5skill_v3_attack_defend.py --rallies $R/$P/rallies_10x.pkl \
      $PHI --seed $s --out-dir $d > $d.log 2>&1 &
done; done
wait
touch $A/TRAIN_DONE

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
export -f eval_one gain_one
export PY A NC SET
{ for P in aggressive moderate_aggressive; do for s in 0 1 2; do
    M=$A/$P/phi/gamma0.99_s$s; SP=$NC/${P}_g099_s$s.pth
    echo "eval_one $P $A/$P/table1/nash-p-pga_s$s nash-p-pga $M --seed $s --episodes 100 --opponents static random ibr nash-p-adaptive selfplay:$SP"
  done; done
  for P in aggressive moderate_aggressive; do for s in 0 1 2; do
    echo "gain_one $P $A/$P/gains/nash-p-pga_s$s nash-p-pga $A/$P/phi/gamma0.99_s$s 2 $s"
  done; done; } | xargs -P 3 -I{} bash -c '{}'
touch $A/ALL_DONE
echo "done $(date -Is)" >> $A/MANIFEST.txt
rm -f $LOCK
