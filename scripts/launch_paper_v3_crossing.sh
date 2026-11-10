#!/bin/bash
# The Design B pipeline rerun with decisions at NET CROSSINGS (decision timing "crossing") instead of racket
# contacts, for both the alpha-potential and self-play. Same state (attack_defend_joint_v3), profile (aggressive),
# defense mechanism (preparation_cal) and training recipes as launch_paper_v3.sh / launch_paper_v3_mixed.sh.
#   data       fixed-pair grid rallies (120 shards) + randomized-theta rallies (120 shards), crossing timing
#   alpha      pure theta: Q/Phi gamma 0.99, 3 seeds -> Eq. 12 (nash-p-argmax), Potential Adaptive Match;
#              mixed theta on grid+randomized data, 3 seeds -> PGA (nash-p-pga)
#   self-play  pi(x, role) without choice slots, crossing timing, 3 seeds
#   evals      Table 1 columns static / random / vs Potential Adaptive Match / the other proposed method,
#              stall cells replayed to the end, Table 2 response gains. No IBR opponents or Q-rule rows.
# Outputs under runs_v3_crossing/ (layout of runs_v3/ for summarize_paper_alpha.py).
set -u
cd ~/angelina/ping-pong-game-290-designb-v3
R=runs_v3_crossing
LOCK=$R.lock
if [ -e $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then echo "already running"; exit 0; fi
echo $$ > $LOCK
PY=~/angelina/envs/designb/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
P=aggressive
A=$R/paper
AM=$R/paper_mixed
ENC=attack_defend_joint_v3
TIMING=crossing
SET="--defend-mechanism preparation_cal --decision-timing $TIMING"
PHI="--theta-mode pure --pure-data-only --horizon 10 --phi-loss lse --phi-select val_residual --q-select val --gamma 0.99"
PHIM="--theta-mode mixed --horizon 10 --phi-loss lse --phi-select val_residual --q-select val --gamma 0.99"
CAP="--max-attempt-factor 3"
mkdir -p $R/data $R/data_mixed $R/selfplay $A/$P/phi $AM/$P/phi
echo "launched $(date -Is) from $(pwd), timing $TIMING" >> $A/MANIFEST.txt

collect_one() { kind=$1; i=$2
  dir=$R/data; base=7000
  if [ "$kind" = mixed ]; then dir=$R/data_mixed; base=8000; fi
  out=$dir/${kind}_$i
  [ -e $out.pkl ] && return 0
  $PY -u nash_skills/v2/collect_data_attack_defend.py --sampling $kind --decisions 1500 --max-rallies 3000 \
      --max-steps 2500 $SET --skill-profile $P --reset-targets --encoder $ENC \
      --seed $((base + i)) --progress-every 0 --output $out.pkl > $out.log 2>&1
}
train_one() { d=$1; rallies=$2; shift 2
  [ -e $d/heldout_attack_defend.json ] && return 0
  OMP_NUM_THREADS=2 $PY -u nash_skills/v2/train_q_model_5skill_v3_attack_defend.py --rallies $rallies "$@" \
      --out-dir $d > $d.log 2>&1
}
eval_one() { out=$1; method=$2; model=$3; shift 3
  mkdir -p $(dirname $out)
  [ -e $out.json ] && return 0
  $PY -u scripts/eval_table1_attack_defend.py $SET --skill-profile $P --workers 8 --method $method \
      --model-dir $model $CAP "$@" --output $out.json > $out.log 2>&1
}
gain_one() { out=$1; method=$2; model=$3; repeats=$4; seed=$5
  mkdir -p $(dirname $out)
  [ -e $out.json ] && return 0
  $PY -u scripts/eval_response_gains.py --method $method --model-dir $model --skill-profile $P \
      --decision-timing $TIMING --points 32 --workers 8 --repeats $repeats --seed $seed --output $out.json > $out.log 2>&1
}
replay_one() { out=$1; method=$2; model=$3; seed=$4
  mkdir -p $(dirname $out)
  [ -e $out.json ] && return 0
  $PY -u scripts/diagnose_unfinished_rallies.py --method $method --model-dir $model --decision-timing $TIMING \
      --opponents center_safe/center center_safe/left --rallies-per-side 30 --max-steps 20000 --workers 4 \
      --seed $((80 + seed)) --output $out.json > $out.log 2>&1
}
export -f collect_one train_one eval_one gain_one replay_one
export PY R A AM P ENC TIMING SET PHI PHIM CAP

# ---- self-play: needs no data, starts at once ---------------------------------------------------------
( for s in 0 1 2; do
    out=$R/selfplay/${P}_s$s
    [ -e $out.done ] && continue
    ( $PY -u nash_skills/v2/selfplay_per_player.py --iterations 1000 --rallies-per-iter 32 --round-iters 25 \
          --defend-mechanism preparation_cal --decision-timing $TIMING --skill-profile $P --reset-targets \
          --gamma 0.99 --seed $s --state-mode no_choices --encoder $ENC --output $out.pth --log $out.csv \
          > $out.log 2>&1 && touch $out.done ) &
  done; wait; touch $R/SELFPLAY_DONE ) &

# ---- data: grid shards, then randomized-theta shards ----------------------------------------------------
seq 0 119 | xargs -P 16 -I{} bash -c 'collect_one grid {}'
[ -e $R/data/rallies.pkl ] || $PY scripts/merge_rally_shards.py --output $R/data/rallies.pkl $R/data/grid_*.pkl \
    > $R/data/merge.log 2>&1
[ -e $R/data/rallies.pkl ] || { echo "grid merge failed" >> $A/MANIFEST.txt; rm -f $LOCK; exit 1; }
touch $R/DATA_DONE

# ---- alpha-potential, pure theta: 3 seeds (the randomized shards collect meanwhile) -----------------------
( for s in 0 1 2; do train_one $A/$P/phi/gamma0.99_s$s $R/data/rallies.pkl $PHI --seed $s & done; wait
  touch $A/TRAIN_DONE ) &
seq 0 119 | xargs -P 16 -I{} bash -c 'collect_one mixed {}'
[ -e $R/data_mixed/rallies_all.pkl ] || $PY scripts/merge_rally_shards.py --output $R/data_mixed/rallies_all.pkl \
    $R/data/grid_*.pkl $R/data_mixed/mixed_*.pkl > $R/data_mixed/merge.log 2>&1
[ -e $R/data_mixed/rallies_all.pkl ] || { echo "mixed merge failed" >> $A/MANIFEST.txt; rm -f $LOCK; exit 1; }
touch $R/DATA_MIXED_DONE
for s in 0 1 2; do train_one $AM/$P/phi/gamma0.99_s$s $R/data_mixed/rallies_all.pkl $PHIM --seed $s & done; wait
until [ -e $A/TRAIN_DONE ]; do sleep 60; done
for s in 0 1 2; do
  for d in $A/$P/phi/gamma0.99_s$s $AM/$P/phi/gamma0.99_s$s; do
    [ -e $d/heldout_attack_defend.json ] || { echo "training failed: $d" >> $A/MANIFEST.txt; rm -f $LOCK; exit 1; }
  done
done
touch $AM/TRAIN_DONE

# ---- evaluations (self-play rows wait for the self-play training) -------------------------------------
until [ -e $R/SELFPLAY_DONE ]; do sleep 300; done
{ for s in 0 1 2; do M=$A/$P/phi/gamma0.99_s$s; MM=$AM/$P/phi/gamma0.99_s$s; SP=$R/selfplay/${P}_s$s.pth
    echo "eval_one $A/$P/table1/nash-p-argmax_s$s nash-p-argmax $M --seed $s --episodes 100 --opponents static random nash-p-adaptive selfplay:$SP"
    echo "eval_one $A/$P/table1/nash-p-adaptive_s$s nash-p-adaptive $M --seed $s --episodes 100 --opponents static random nash-p-adaptive selfplay:$SP"
    echo "eval_one $AM/$P/table1/nash-p-pga_s$s nash-p-pga $MM --seed $s --episodes 100 --opponents static random nash-p-adaptive selfplay:$SP"
    for kind in selfplay selfplay-argmax; do n=${kind/selfplay/selfplay-nc}
      echo "eval_one $A/$P/table1/${n}_s$s $kind:$SP $M --seed $s --episodes 100 --opponents static random nash-p-adaptive nash-p-argmax"
    done
    echo "replay_one $A/$P/diag_unfinished/nash-p-argmax_s$s nash-p-argmax $M $s"
    echo "replay_one $A/$P/diag_unfinished/nash-p-adaptive_s$s nash-p-adaptive $M $s"
    echo "replay_one $AM/$P/diag_unfinished/nash-p-pga_s$s nash-p-pga $MM $s"
    echo "replay_one $A/$P/diag_unfinished/selfplay-nc_s$s selfplay:$SP $M $s"
    echo "replay_one $A/$P/diag_unfinished/selfplay-nc-argmax_s$s selfplay-argmax:$SP $M $s"
    echo "gain_one $A/$P/gains/nash-p-argmax_s$s nash-p-argmax $M 1 $s"
    echo "gain_one $A/$P/gains/nash-p-adaptive_s$s nash-p-adaptive $M 1 $s"
    echo "gain_one $AM/$P/gains/nash-p-pga_s$s nash-p-pga $MM 2 $s"
    echo "gain_one $A/$P/gains/selfplay-nc_s$s selfplay:$SP $M 3 $s"
    echo "gain_one $A/$P/gains/selfplay-nc-argmax_s$s selfplay-argmax:$SP $M 1 $s"
  done; } | xargs -P 4 -I{} bash -c '{}'
touch $R/ALL_DONE
echo "done $(date -Is)" >> $A/MANIFEST.txt
rm -f $LOCK
