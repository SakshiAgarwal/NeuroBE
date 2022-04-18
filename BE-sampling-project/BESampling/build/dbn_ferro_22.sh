#!/bin/bash

#Parameters of our algorithm in general
export CUDA_VISIBLE_DEVICES=3


learning_rate="0.001"	# Optimization learning rate
n_epochs="500"					# Number of epochs to train the net
batch_size="256"			# Number of training sequences in each batch
network="net"
s_method="is"
stop_iter="2"
width_problem="20"
dim="3"
e="0.1"

input_instance_folder="/home/sakshia1/myresearch/GDBE/BucketElimination-mac-version/BE-sampling-project/3testproblems/rbm_ferro_22.uai"
input_ordering_folder="/home/sakshia1/myresearch/GDBE/BucketElimination-mac-version/BE-sampling-project/3testproblems/rbm_ferro_22.uai.ord.elim"
results_prefix="/home/sakshia1/myresearch/GDBE/BucketElimination-mac-version-local-copy/results/aistats/rbm_ferro_22_NeuroBE-dim-"$dim"-epsilon-"
i="1 2 3 4 5"

for j in $i;do	
	for s_m in $s_method; do
		       for epsilon in $e;do
			       for var_dim in $dim;do
				       results_prefix="/home/sakshia1/myresearch/GDBE/BucketElimination-mac-version-local-copy/results/aistats/rbm_ferro_22_iter_NeuroBE-dim-"$dim"-epsilon-"
			       		./BESampling -fUAI $input_instance_folder -fVO $input_ordering_folder -iB 999 -v2sample 369 -nsamples 10 -batch_size $batch_size -lr $learning_rate -n_epochs $n_epochs --network $network --out_file ${results_prefix}${epsilon}_${var_dim}_${s_m} --sampling_method $s_m --width_problem $width_problem --stop_iter $stop_iter --var_dim $var_dim --epsilon $epsilon
		       		done
			done
	done
done





