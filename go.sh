#!/bin/bash -ex

model_name="my_model"
for seed in 2 5 8 16 25
do
  python train.py --seed $seed --model_name $model_name
  cd testing_final_20201012
  python test_seq2graph.py --seed $seed --model_name $model_name
  cd ../evaluation
  python run.py --seed $seed --model_name $date
  cd ..
done
