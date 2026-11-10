#!/bin/bash
# Self-play without the choice slots in the actor input (Dvij's pi(s1, s2, role)), same settings as
# runs_paper/selfplay otherwise; then Table 1 rows for the new runs, and the old runs re-evaluated
# against the same IBR / alpha-potential opponents (fixed-Q models) for a like-for-like comparison.
set -u
cd ~/angelina/ping-pong-game-290-designb
LOCK=runs_paper/selfplay_nochoice.lock
if [ -e $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then echo "already running"; exit 0; fi
echo $$ > $LOCK
PY=~/angelina/envs/designb/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
R=runs_paper
N=$R/selfplay_nochoice
mkdir -p $N
echo "launched $(date -Is)" >> $N/MANIFEST.txt
pids=()
for P in aggressive moderate_aggressive; do
  for s in 0 1 2; do
    [ -e $N/${P}_g099_s$s.done ] && continue
    ( $PY -u nash_skills/v2/selfplay_per_player.py --iterations 1000 --rallies-per-iter 32 --round-iters 25 \
          --defend-mechanism preparation_cal --skill-profile $P --reset-targets --gamma 0.99 --seed $s \
          --state-mode no_choices --output $N/${P}_g099_s$s.pth --log $N/${P}_g099_s$s.csv \
          > $N/${P}_g099_s$s.log 2>&1 && touch $N/${P}_g099_s$s.done ) &
    pids+=($!)
  done
done
wait "${pids[@]}"
touch $N/TRAIN_DONE
run_one() { P=$1; version=$2; kind=$3; s=$4
  if [ $version = new ]; then CK=$N/${P}_g099_s$s.pth; OPP="static random ibr nash-p-pga"
  else CK=$R/selfplay/${P}_g099_s$s.pth; OPP="ibr nash-p-pga"; fi
  out=$N/eval/$P/${version}_${kind}_s$s; mkdir -p $(dirname $out)
  [ -e $out.json ] && return 0
  $PY -u scripts/eval_table1_attack_defend.py --defend-mechanism preparation_cal --decision-timing contact \
      --skill-profile $P --workers 10 --method $kind:$CK --model-dir $R/ablations_qval/$P/phi/gamma0.99_s$s \
      --seed $s --episodes 100 --opponents $OPP --output $out.json > $out.log 2>&1
}
export -f run_one
export PY N R
for P in aggressive moderate_aggressive; do for v in new old; do for k in selfplay selfplay-argmax; do for s in 0 1 2; do
  echo "$P $v $k $s"; done; done; done; done | xargs -P 4 -L 1 bash -c 'run_one "$@"' _
touch $N/ALL_DONE
echo "done $(date -Is)" >> $N/MANIFEST.txt
rm -f $LOCK
