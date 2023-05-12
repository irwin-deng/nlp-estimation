#!/bin/bash

gpu_id=0

for i in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train_model.py --source-dataset "cifar10" --model-type dann_arch --base-dir checkpoints/ensemble_dann_arch_source_models/$i/ --seed $(( 100*(i+1) ))
done
