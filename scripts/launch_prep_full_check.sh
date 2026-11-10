#!/bin/bash
# Does calibrating the preparation orientation (preparation_full) make preparing help?
# One-sided (matched / mismatched / none) and both-sided checks, preparation_cal vs preparation_full.
set -u
cd ~/angelina/ping-pong-game-290-designb
LOCK=runs_paper/prep_full_check2.lock
if [ -e $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then echo "already running"; exit 0; fi
echo $$ > $LOCK
PY=~/angelina/envs/designb/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
C=runs_paper/checks_full2
mkdir -p $C
for P in aggressive moderate_aggressive; do
  ( $PY -u scripts/compare_defend_mechanisms.py --mode one-sided --pairs 30 --workers 12 --skill-profile $P \
        --mechanisms preparation_cal preparation_full --output $C/one_sided_$P.json > $C/one_sided_$P.log 2>&1
    $PY -u scripts/compare_defend_mechanisms.py --mode both-sided --pairs 200 --workers 12 --skill-profile $P \
        --mechanisms preparation_cal preparation_full --output $C/both_sided_$P.json > $C/both_sided_$P.log 2>&1 ) &
done
wait
touch $C/DONE
rm -f $LOCK
