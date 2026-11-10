#!/bin/bash
# The paper's full joint state x_t = (ball incl. spin, both robots, role, rally status): state encoder
# attack_defend_joint_v3, aggressive profile only, everything the paper reports rerun from scratch.
#   data       fixed-pair (Eq. 7) rallies with grid coverage of the 225 profiles, v3 states
#   alpha      Phi/Q as launch_paper_alpha.sh (H = 10, smooth-max loss, validation checkpoints):
#              gamma sweep, nested data volumes; online phase Eq. 12 (nash-p-argmax)
#   self-play  pi(x, role) without the choice slots (Dvij), v3 states, PPO, 3 seeds
#   evals      Table 1 (six rules, alpha-potential, self-play), Table 2 gains, Table 3, simulated IBR
#              (H_IBR = 5) for every row, figure evals; stalled rallies re-drawn at most 3x per task
# Runs from its own copy of the code (~/angelina/ping-pong-game-290-designb-v3), so jobs started from
# the main copy are unaffected. Output layout matches runs_paper/paper_alpha for summarize_paper_alpha.py.
set -u
cd ~/angelina/ping-pong-game-290-designb-v3
LOCK=runs_v3.lock
if [ -e $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then echo "already running"; exit 0; fi
echo $$ > $LOCK
PY=~/angelina/envs/designb/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
P=aggressive
R=runs_v3
A=$R/paper
ENC=attack_defend_joint_v3
SET="--defend-mechanism preparation_cal --decision-timing contact"
PHI="--theta-mode pure --pure-data-only --horizon 10 --phi-loss lse --phi-select val_residual --q-select val"
RULES="nash-q br-q nash-p-ibr nash-p-br nash-p-adaptive nash-p-minimax nash-p-argmax"
CAP="--max-attempt-factor 3"
mkdir -p $R/data $R/selfplay $A/$P/phi
echo "launched $(date -Is) from $(pwd)" >> $A/MANIFEST.txt

collect_one() { i=$1; out=$R/data/grid_$i
  [ -e $out.pkl ] && return 0
  $PY -u nash_skills/v2/collect_data_attack_defend.py --sampling grid --decisions 1500 --max-rallies 3000 \
      --max-steps 2500 $SET --skill-profile $P --reset-targets --encoder $ENC --seed $((5000 + i)) \
      --progress-every 0 --output $out.pkl > $out.log 2>&1
}
train_one() { name=$1; shift
  d=$A/$P/phi/$name
  [ -e $d/heldout_attack_defend.json ] && return 0
  OMP_NUM_THREADS=2 $PY -u nash_skills/v2/train_q_model_5skill_v3_attack_defend.py --rallies $R/data/rallies.pkl \
      $PHI "$@" --out-dir $d > $d.log 2>&1
}
eval_one() { out=$1; method=$2; model=$3; workers=$4; shift 4
  mkdir -p $(dirname $out)
  [ -e $out.json ] && return 0
  $PY -u scripts/eval_table1_attack_defend.py $SET --skill-profile $P --workers $workers --method $method \
      --model-dir $model $CAP "$@" --output $out.json > $out.log 2>&1
}
gain_one() { out=$1; method=$2; model=$3; repeats=$4; seed=$5
  mkdir -p $(dirname $out)
  [ -e $out.json ] && return 0
  $PY -u scripts/eval_response_gains.py --method $method --model-dir $model --skill-profile $P \
      --points 32 --workers 8 --repeats $repeats --seed $seed --output $out.json > $out.log 2>&1
}
# rollouts per candidate for the response gains: stochastic methods need more (as in launch_paper_alpha.sh)
repeats_for() { case $1 in nash-q|selfplay) echo 3 ;; br-q) echo 2 ;; *) echo 1 ;; esac; }
export -f collect_one train_one eval_one gain_one repeats_for
export PY R A P ENC SET PHI CAP

# ---- self-play: needs no data, starts at once ---------------------------------------------------------
( for s in 0 1 2; do
    out=$R/selfplay/${P}_s$s
    [ -e $out.done ] && continue
    ( $PY -u nash_skills/v2/selfplay_per_player.py --iterations 1000 --rallies-per-iter 32 --round-iters 25 \
          --defend-mechanism preparation_cal --skill-profile $P --reset-targets --gamma 0.99 --seed $s \
          --state-mode no_choices --encoder $ENC --output $out.pth --log $out.csv > $out.log 2>&1 \
          && touch $out.done ) &
  done; wait; touch $R/SELFPLAY_DONE ) &

# ---- data: 120 shards x 1500 decisions (~20k fixed-pair rallies, as in runs_paper) ----------------------
seq 0 119 | xargs -P 16 -I{} bash -c 'collect_one {}'
[ -e $R/data/rallies.pkl ] || $PY scripts/merge_rally_shards.py --output $R/data/rallies.pkl $R/data/grid_*.pkl \
    > $R/data/merge.log 2>&1
touch $R/DATA_DONE

# ---- alpha-potential: gamma sweep and nested data volumes ---------------------------------------------
{ for s in 0 1 2; do
    for g in 0.5 0.6 0.7 0.8 0.9 0.95 0.99 1.0; do echo "train_one gamma${g}_s$s --gamma $g --seed $s"; done
    for n in 170 550 1700 5500; do echo "train_one vol${n}_s$s --gamma 0.99 --seed $s --train-rally-limit $n"; done
  done; } | xargs -P 12 -I{} bash -c '{}'
for s in 0 1 2; do
  N=$($PY -c "import json; print(json.load(open('$A/$P/phi/gamma0.99_s$s/heldout_attack_defend.json'))['n_train_rallies'])")
  ( cd $A/$P/phi && ln -sfn gamma0.99_s$s vol${N}_s$s )
done
touch $A/TRAIN_DONE

# ---- self-play evaluations wait for both trainings ----------------------------------------------------
( until [ -e $R/SELFPLAY_DONE ]; do sleep 300; done
  { for s in 0 1 2; do M=$A/$P/phi/gamma0.99_s$s; SP=$R/selfplay/${P}_s$s.pth
      for kind in selfplay selfplay-argmax; do n=${kind/selfplay/selfplay-nc}
        echo "eval_one $A/$P/table1/${n}_s$s $kind:$SP $M 8 --seed $s --episodes 100 --opponents static random ibr nash-p-adaptive nash-p-argmax"
        echo "gain_one $A/$P/gains/${n}_s$s $kind:$SP $M \$(repeats_for $kind) $s"
        echo "eval_one $A/$P/ibr_sim/${n}_s$s $kind:$SP $M 8 --seed $((70 + s)) --episodes 40 --opponents ibr-sim --split ibr-sim=20 --ibr-horizon 5"
      done
    done; } | xargs -P 3 -I{} bash -c '{}'
  touch $A/SELFPLAY_EVAL_DONE ) &

# ---- alpha-potential and rule evaluations: Table 1, gains, figure evals, simulated IBR --------------------
{ for s in 0 1 2; do M=$A/$P/phi/gamma0.99_s$s
    for rule in $RULES; do
      echo "eval_one $A/$P/table1/${rule}_s$s $rule $M 8 --seed $s --episodes 100 --opponents static random ibr nash-p-adaptive"
    done
  done
  for s in 0 1 2; do M=$A/$P/phi/gamma0.99_s$s
    for rule in $RULES; do echo "gain_one $A/$P/gains/${rule}_s$s $rule $M \$(repeats_for $rule) $s"; done
  done
  for s in 0 1 2; do
    for d in $A/$P/phi/gamma*_s$s $A/$P/phi/vol*_s$s; do n=$(basename $d)
      echo "eval_one $A/$P/figevals/$n nash-p-argmax $d 8 --seed $s --episodes 60 --opponents static random"
    done
  done
  for s in 0 1 2; do M=$A/$P/phi/gamma0.99_s$s
    for rule in $RULES; do
      echo "eval_one $A/$P/ibr_sim/${rule}_s$s $rule $M 8 --seed $((70 + s)) --episodes 40 --opponents ibr-sim --split ibr-sim=20 --ibr-horizon 5"
    done
  done; } | xargs -P 4 -I{} bash -c '{}'
touch $A/ALPHA_EVAL_DONE
wait
touch $A/ALL_DONE
echo "done $(date -Is)" >> $A/MANIFEST.txt
rm -f $LOCK
