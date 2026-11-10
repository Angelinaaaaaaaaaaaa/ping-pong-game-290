#!/bin/bash
# IBR-Q (best response by own Q to the opponent's empirical choice frequencies over the match) on the
# base and swap-augmented alpha-potential models; 3 seeds x 2 profiles; same opponents as kexin_format.
set -u
cd ~/angelina/ping-pong-game-290-designb
LOCK=runs_paper/ibrq.lock
if [ -e $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then echo "already running"; exit 0; fi
echo $$ > $LOCK
PY=~/angelina/envs/designb/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
R=runs_paper
run_one() { P=$1; variant=$2; s=$3
  if [ $variant = base ]; then D=$R/ablations_qval/$P/phi/gamma0.99_s$s; else D=$R/swap_aug/$P/phi/g099_s$s; fi
  out=$R/ibrq/$P/${variant}_s$s; mkdir -p $(dirname $out)
  [ -e $out.json ] && return 0
  $PY -u scripts/eval_table1_attack_defend.py --defend-mechanism preparation_cal --decision-timing contact \
      --skill-profile $P --workers 10 --method ibr-q --model-dir $D --seed $s --episodes 100 \
      --opponents static random ibr selfplay:$R/selfplay/${P}_g099_s$s.pth --output $out.json > $out.log 2>&1
}
export -f run_one
export PY R
for P in aggressive moderate_aggressive; do for v in base swap; do for s in 0 1 2; do echo "$P $v $s"; done; done; done \
  | xargs -P 4 -L 1 bash -c 'run_one "$@"' _
touch $R/ibrq/DONE
rm -f $LOCK
