#!/bin/bash
# alpha-potential ablations for the paper plots (both profiles, 3 seeds, Table 1 static + random):
#   gamma in {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0} on the base data set (rallies.pkl)
#   data volume: nested train subsets of the 10x set (rallies_10x.pkl), gamma 0.99,
#                ~1/10, 1/3, 1x, 3x, 10x of the base training size
# Usage: QSEL=val bash scripts/launch_alpha_ablations.sh   (QSEL: Q selection, last | val)
set -u
cd ~/angelina/ping-pong-game-290-designb
PY=~/angelina/envs/designb/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
R=runs_paper
A=$R/ablations_q${QSEL:-val}
SET="--defend-mechanism preparation_cal --decision-timing contact"
PHI_OPTS="--theta-mode mixed --phi-loss lse --phi-select val_residual --q-select ${QSEL:-val}"
mkdir -p $A
echo "launched $(date -Is) QSEL=${QSEL:-val}" >> $A/MANIFEST.txt

phi_jobs() {  # one line per Phi run: profile name rallies extra-args...
  for P in aggressive moderate_aggressive; do
    for g in 0.5 0.6 0.7 0.8 0.9 0.95 0.99 1.0; do
      for s in 0 1 2; do echo "$P gamma${g}_s$s $R/$P/rallies.pkl --gamma $g --seed $s"; done
    done
  done
}
vol_jobs() {
  for P in aggressive moderate_aggressive; do
    N=$($PY -c "import pickle; print(len(pickle.load(open('$R/$P/rallies_10x.pkl','rb'))))")
    FULL=$((N * 72 / 100))  # 80% train, then 10% of that for validation
    for n in 340 1100 3400 11000 $FULL; do
      for s in 0 1 2; do echo "$P vol${n}_s$s $R/$P/rallies_10x.pkl --gamma 0.99 --seed $s --train-rally-limit $n"; done
    done
  done
}
run_phi() {  # $1 profile $2 name $3 rallies, rest: args
  P=$1; name=$2; rallies=$3; shift 3
  mkdir -p $A/$P/phi
  [ -e $A/$P/phi/$name/heldout_attack_defend.json ] && return 0
  OMP_NUM_THREADS=2 $PY -u nash_skills/v2/train_q_model_5skill_v3_attack_defend.py --rallies $rallies $PHI_OPTS "$@" \
      --out-dir $A/$P/phi/$name > $A/$P/phi/$name.log 2>&1
}
run_eval() {  # $1 profile $2 name
  P=$1; name=$2; seed=${name##*_s}
  mkdir -p $A/$P/table1
  [ -e $A/$P/table1/$name.json ] && return 0
  $PY -u scripts/eval_table1_attack_defend.py $SET --skill-profile $P --workers 10 --method nash-p-pga \
      --model-dir $A/$P/phi/$name --seed $seed --episodes 60 --opponents static random \
      --output $A/$P/table1/$name.json > $A/$P/table1/$name.log 2>&1
}
export -f run_phi run_eval
export PY A R SET PHI_OPTS

# gamma sweep: train all, then evaluate (4 evaluations x 10 workers at a time)
phi_jobs | xargs -P 16 -L 1 bash -c 'run_phi "$@"' _
phi_jobs | awk '{print $1, $2}' | xargs -P 4 -L 1 bash -c 'run_eval "$@"' _
touch $A/GAMMA_DONE

# data volume, once the 10x data exists
until [ -e $R/DATA10X_DONE ]; do sleep 120; done
vol_jobs | xargs -P 12 -L 1 bash -c 'run_phi "$@"' _
vol_jobs | awk '{print $1, $2}' | xargs -P 4 -L 1 bash -c 'run_eval "$@"' _
touch $A/ALL_DONE
echo "done $(date -Is)" >> $A/MANIFEST.txt
