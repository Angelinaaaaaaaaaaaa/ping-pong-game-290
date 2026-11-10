#!/bin/bash
# Kexin-format tables: every V3 decision rule on our alpha-potential models (fixed Q, gamma 0.99),
# 3 seeds x 2 profiles x {base data (gamma0.99_s*), 10x data (vol3xxxx_s*)}.
# Opponents: 15 static, random, Q-based IBR, self-play (sampled, same seed); 100 decided rallies per side.
# The base-data PGA row already exists in runs_paper/alpha_fixed (same protocol).
set -u
cd ~/angelina/ping-pong-game-290-designb
LOCK=runs_paper/kexin_format.lock
if [ -e $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then echo "already running (pid $(cat $LOCK))"; exit 0; fi
echo $$ > $LOCK
PY=~/angelina/envs/designb/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
R=runs_paper
A=$R/ablations_qval
K=$R/kexin_format
mkdir -p $K
echo "launched $(date -Is)" >> $K/MANIFEST.txt
RULES="nash-q br-q nash-p-ibr nash-p-br nash-p-adaptive nash-p-minimax"

job_list() {
  for data in base x10; do
    for P in aggressive moderate_aggressive; do
      for s in 0 1 2; do
        if [ $data = base ]; then D=$A/$P/phi/gamma0.99_s$s; rules=$RULES
        else D=$(ls -d $A/$P/phi/vol3[0-9][0-9][0-9][0-9]_s$s | head -1); rules="nash-p-pga $RULES"; fi
        for rule in $rules; do echo "$P $data $s $rule $D"; done
      done
    done
  done
}
run_one() {
  P=$1; data=$2; s=$3; rule=$4; D=$5
  out=$K/$P/$data/${rule}_s$s
  mkdir -p $(dirname $out)
  [ -e $out.json ] && return 0
  $PY -u scripts/eval_table1_attack_defend.py --defend-mechanism preparation_cal --decision-timing contact \
      --skill-profile $P --workers 10 --method $rule --model-dir $D --seed $s --episodes 100 \
      --opponents static random ibr selfplay:$R/selfplay/${P}_g099_s$s.pth \
      --output $out.json > $out.log 2>&1
}
export -f run_one
export PY K R
job_list | xargs -P 4 -L 1 bash -c 'run_one "$@"' _
touch $K/ALL_DONE
echo "done $(date -Is)" >> $K/MANIFEST.txt
rm -f $LOCK
