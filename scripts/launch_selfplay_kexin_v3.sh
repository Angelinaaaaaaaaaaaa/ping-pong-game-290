#!/bin/bash
# Route B: self-play in Kexin's own V3 game, scored with her own evaluator.
#  1. snapshot of her checkout (code, simulator assets, controller, her 50k Q and Phi models; her working tree
#     as is, uncommitted simulator edits included) in ~/angelina/kexin-v3-snapshot, plus our v3_selfplay package
#     (rsynced there beforehand);
#  2. self-play training, 3 seeds (v3_selfplay.train: our Design B self-play schedule, her state and protocol);
#  3. her Table 1 protocol (v3/eval_policy_fixed_grid.py: 15 static, random, IBR = greedy_q, other proposed =
#     Potential Adaptive Match) for self-play (sampled and argmax), and the same run for her Potential Adaptive
#     Match as a check against her reported numbers;
#  4. head-to-head of self-play against her potential rules, Nash-Q and BR-Q.
# Outputs in ~/angelina/kexin-v3-snapshot/runs_selfplay_v3.
set -u
SNAP=~/angelina/kexin-v3-snapshot
PY=~/angelina/envs/designb/bin/python
bash ~/angelina/selfplay_kexin_v3/snapshot_kexin_v3.sh
cd $SNAP
LOCK=$SNAP/selfplay_v3.lock
if [ -e $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then echo "already running"; exit 0; fi
echo $$ > $LOCK
O=runs_selfplay_v3
mkdir -p $O
echo "launched $(date -Is)" >> $O/MANIFEST.txt
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
Q=models/v3/q_preposition_onehot_50k_aggressive.pth
PHI=models/v3/potential_50k_aggressive.pth
EPISODES=100
OPPONENTS="potential_ibr potential_br potential_adaptive_match potential_minimax nash_q br_q"

grid_one() { out=$1; policy=$2; seed=$3; shift 3
  [ -e $out/batch_policy_eval_summary.json ] && return 0
  mkdir -p $out
  $PY -u -m v3_selfplay.her_eval grid "$@" -- --policies $policy --model $Q --potential-model $PHI \
      --seeds $seed --episodes $EPISODES --both-sides --steps 600 --output-dir $out > $out.log 2>&1
}
h2h_one() { out=$1; seed=$2; shift 2
  [ -e $out.json ] && return 0
  $PY -u -m v3_selfplay.her_eval h2h "$@" --opponents $OPPONENTS --model $Q --potential-model $PHI \
      --seeds $seed --episodes $EPISODES --both-sides --output-json $out.json > $out.log 2>&1
}
export -f grid_one h2h_one
export PY Q PHI EPISODES OPPONENTS

# her own Potential Adaptive Match under the same settings, while self-play trains
( grid_one $O/grid_kexin_pam_s0 potential_adaptive_match 0 ) &

for s in 0 1 2; do
  ( [ -e $O/aggressive_s$s.done ] || { $PY -u -m v3_selfplay.train --seed $s --workers 8 --output $O/aggressive_s$s.pth \
        --log $O/aggressive_s$s.csv > $O/aggressive_s$s.log 2>&1 && touch $O/aggressive_s$s.done; } ) &
done
wait
echo "training done $(date -Is)" >> $O/MANIFEST.txt

{ for s in 0 1 2; do C=$O/aggressive_s$s.pth
    echo "grid_one $O/grid_sp_s$s sp $s --register sp=$C"
    echo "grid_one $O/grid_sp_argmax_s$s sp_argmax $s --register sp_argmax=$C:argmax"
    echo "h2h_one $O/h2h_sp_s$s $s --register sp=$C --tested sp"
    echo "h2h_one $O/h2h_sp_argmax_s$s $s --register sp_argmax=$C:argmax --tested sp_argmax"
  done; } | xargs -P 12 -I{} bash -c '{}'
touch $O/ALL_DONE
echo "done $(date -Is)" >> $O/MANIFEST.txt
rm -f $LOCK
