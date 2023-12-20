#!/bin/bash -ex

date="20230628"
for seed in 2 5 8 16 25
do
  python run.py --seed $seed --model_name $date --chat True
done
