#!/bin/bash


runs=5
exps=(muta recomb_random recomb_layerwise)

for exp in ${exps[@]}; do
    for seed in $(seq 0 $(($runs-1))); do
        echo "starting $exp-$seed"
        tmux new-session -t "$exp-$seed" -d
        tmux send-keys -t "$exp-$seed" "scripts/exp.sh $exp $seed" C-m
        #tmux kill-session -t "$exp-$seed"
    done
done
