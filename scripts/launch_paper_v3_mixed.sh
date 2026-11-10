#!/bin/bash
# Randomised policy parameters on the new state (attack_defend_joint_v3, aggressive): each player's theta is a
# distribution over the 5 targets and one over the 3 preparations instead of paper Eq. 7's single pair.
#   data   randomised-theta rallies (collector --sampling mixed: one theta per player per rally, pure with
#          probability 0.3), merged with the fixed-pair rallies of launch_paper_v3.sh
#   train  Q/Phi as the paper protocol (H = 10, smooth-max loss, validation checkpoints), theta-mode mixed, 3 seeds
#   evals  online projected gradient ascent on Phi (nash-p-pga): Table 1 opponents plus self-play, the stall cell
#          played to the end, response gains, simulated IBR
set -u
cd ~/angelina/ping-pong-game-290-designb-v3
LOCK=runs_v3/paper_mixed.lock
if [ -e $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then echo "already running"; exit 0; fi
echo $$ > $LOCK
PY=~/angelina/envs/designb/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
P=aggressive
R=runs_v3
A=$R/paper_mixed
ENC=attack_defend_joint_v3
SET="--defend-mechanism preparation_cal --decision-timing contact"
CAP="--max-attempt-factor 3"
PHI="--theta-mode mixed --horizon 10 --phi-loss lse --phi-select val_residual --q-select val --gamma 0.99"
mkdir -p $R/data_mixed $A/$P/phi
echo "launched $(date -Is)" >> $A/MANIFEST.txt

collect_one() { i=$1; out=$R/data_mixed/mixed_$i
  [ -e $out.pkl ] && return 0
  $PY -u nash_skills/v2/collect_data_attack_defend.py --sampling mixed --decisions 1500 --max-rallies 3000 \
      --max-steps 2500 $SET --skill-profile $P --reset-targets --encoder $ENC --seed $((6000 + i)) \
      --progress-every 0 --output $out.pkl > $out.log 2>&1
}
export -f collect_one
export PY R P ENC SET
seq 0 119 | xargs -P 20 -I{} bash -c 'collect_one {}'
[ -e $R/data_mixed/rallies_all.pkl ] || $PY scripts/merge_rally_shards.py --output $R/data_mixed/rallies_all.pkl \
    $R/data/grid_*.pkl $R/data_mixed/mixed_*.pkl > $R/data_mixed/merge.log 2>&1
[ -e $R/data_mixed/rallies_all.pkl ] || { echo "merge failed" >> $A/MANIFEST.txt; rm -f $LOCK; exit 1; }
touch $A/DATA_DONE

for s in 0 1 2; do
  d=$A/$P/phi/gamma0.99_s$s
  [ -e $d/heldout_attack_defend.json ] && continue
  OMP_NUM_THREADS=2 $PY -u nash_skills/v2/train_q_model_5skill_v3_attack_defend.py --rallies $R/data_mixed/rallies_all.pkl \
      $PHI --seed $s --out-dir $d > $d.log 2>&1 &
done
wait
for s in 0 1 2; do
  [ -e $A/$P/phi/gamma0.99_s$s/heldout_attack_defend.json ] || { echo "training failed s$s" >> $A/MANIFEST.txt; rm -f $LOCK; exit 1; }
done
touch $A/TRAIN_DONE

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
      --points 32 --workers 8 --repeats 2 --seed $seed --output $out.json > $out.log 2>&1
}
replay_one() { out=$1; method=$2; model=$3; seed=$4
  mkdir -p $(dirname $out)
  [ -e $out.json ] && return 0
  $PY -u scripts/diagnose_unfinished_rallies.py --method $method --model-dir $model \
      --opponents center_safe/center --rallies-per-side 30 --max-steps 20000 --workers 4 --seed $((80 + seed)) \
      --output $out.json > $out.log 2>&1
}
export -f eval_one gain_one replay_one
export A CAP
{ for s in 0 1 2; do M=$A/$P/phi/gamma0.99_s$s; SP=$R/selfplay/${P}_s$s.pth
    echo "eval_one $A/$P/table1/nash-p-pga_s$s nash-p-pga $M --seed $s --episodes 100 --opponents static random ibr nash-p-adaptive selfplay:$SP"
    echo "replay_one $A/$P/diag_unfinished/nash-p-pga_s$s nash-p-pga $M $s"
    echo "gain_one $A/$P/gains/nash-p-pga_s$s nash-p-pga $M $s"
  done
  for s in 0 1 2; do
    echo "eval_one $A/$P/ibr_sim/nash-p-pga_s$s nash-p-pga $A/$P/phi/gamma0.99_s$s --seed $((70 + s)) --episodes 40 --opponents ibr-sim --split ibr-sim=20 --ibr-horizon 5"
  done; } | xargs -P 4 -I{} bash -c '{}'
touch $A/ALL_DONE
echo "done $(date -Is)" >> $A/MANIFEST.txt
rm -f $LOCK
