#!/bin/bash
# Route A: self-play against Kexin's V3 alpha-potential inside our Design B game (decisions at racket contacts,
# calibrated preparation poses). Her player is nash_skills.kexin_v3_player: her 5 x 3 Phi network
# (potential_50k_aggressive.pth, copied into models_kexin/) with each of her four potential rules. Also her rules
# against the 15 static opponents and random, to show how her player fares in this game on its own.
# Outputs in runs_v3/vs_kexin.
set -u
cd ~/angelina/ping-pong-game-290-designb-v3
LOCK=runs_v3/vs_kexin.lock
if [ -e $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then echo "already running"; exit 0; fi
echo $$ > $LOCK
PY=~/angelina/envs/designb/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
P=aggressive
R=runs_v3
A=$R/vs_kexin
PHI=models_kexin/v3/potential_50k_aggressive.pth
RULES="potential_ibr potential_br potential_adaptive_match potential_minimax"
SET="--defend-mechanism preparation_cal --decision-timing contact"
CAP="--max-attempt-factor 3"
mkdir -p $A
echo "launched $(date -Is) phi=$PHI" >> $A/MANIFEST.txt
OPPS=""
for r in $RULES; do OPPS="$OPPS kexin-v3:$r:$PHI"; done

eval_one() { out=$1; method=$2; shift 2
  [ -e $out.json ] && return 0
  $PY -u scripts/eval_table1_attack_defend.py $SET --skill-profile $P --workers 8 --method $method $CAP "$@" \
      --output $out.json > $out.log 2>&1
}
export -f eval_one
export PY P SET CAP
{ for s in 0 1 2; do SP=$R/selfplay/${P}_s$s.pth
    for kind in selfplay selfplay-argmax; do n=${kind/selfplay/selfplay-nc}
      echo "eval_one $A/${n}_s$s $kind:$SP --seed $s --episodes 100 --opponents$OPPS"
    done
  done
  for r in $RULES; do
    echo "eval_one $A/kexin_${r}_static kexin-v3:$r:$PHI --seed 0 --episodes 50 --opponents static random"
  done; } | xargs -P 3 -I{} bash -c '{}'
touch $A/ALL_DONE
echo "done $(date -Is)" >> $A/MANIFEST.txt
rm -f $LOCK
