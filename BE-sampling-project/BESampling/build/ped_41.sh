#!/bin/bash

export CUDA_VISIBLE_DEVICES=0


# Hyper-parameters of the NN
learning_rate="0.001"   # Optimization learning rate
n_epochs="500"                                  # Number of epochs to train the net
batch_size="256"                        # Number of training sequences in each batch
network="masked_net"
s_method="is"
stop_iter="2"
width_problem="20"

input_instance_folder="/home/sakshia1/myresearch/GDBE/BucketElimination-mac-version/BE-sampling-project/3testproblems/pedigree41.uai"
input_ordering_folder="/home/sakshia1/myresearch/GDBE/BucketElimination-mac-version/BE-sampling-project/3testproblems/pedigree41.uai.ord.elim"
results_prefix="/home/sakshia1/myresearch/GDBE/BucketElimination-mac-version-local-copy/results/aistats/ped41_NeuroBE"
i="1 2 3 4 5"

n_layers="2"
epsilon="0.1"
var_dim="3"

for j in $i;do

	./BESampling -fUAI $input_instance_folder -fVO $input_ordering_folder -iB 999 -v2sample 369 -nsamples 10 -batch_size $batch_size -lr $learning_rate -n_epochs $n_epochs --network $network --out_file ${results_prefix}${epsilon}_${var_dim} --out_file2 ${results_prefix}${epsilon}_${var_dim} --sampling_method $s_method --width_problem $width_problem --stop_iter $stop_iter  --n_layers $n_layers --var_dim $var_dim --epsilon $epsilon t

done


