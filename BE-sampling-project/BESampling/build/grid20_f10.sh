#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# Hyper-parameters of the NN
learning_rate="0.001"		# Optimization learning rate
n_epochs="500"				# Maximum number of epochs to train the net
batch_size="256"			# Number of training sequences in each batch
network="net"				# using net "net" or masked net "masked-net"
s_method="is"				# Whether loss is a weighted loss "is" or uniform "us"
width_problem="10"			# width for the problem, training occurs for buckets >width_problem
stop_iter="2"				# condition to stop training when val error increases cont. for more than "stop_iter" times

# Variability in NNs
dim="1"					# vary the hidden dimensions of the neural net
e="0.35"					# needed to determine #samples. Eq: nSamples = int((pd + log(1/delta))/global_config.epsilon);

input_instance_folder="/home/sakshia1/myresearch/GDBE/BucketElimination-mac-version/BE-sampling-project/3testproblems/grid20x20.f10.uai"
input_ordering_folder="/home/sakshia1/myresearch/GDBE/BucketElimination-mac-version/BE-sampling-project/3testproblems/grid20x20.f10.uai.ord.elim"
results_prefix="/home/sakshia1/myresearch/GDBE/BucketElimination-mac-version-local-copy/results/grid20x20.f10"

i="1 2 3 4 5"

for j in $i; do
	for epsilon in $e;do
		for s_m in $s_method;do
			./BESampling -fUAI $input_instance_folder -fVO $input_ordering_folder -iB 999 -v2sample 369 -nsamples 10 -batch_size $batch_size -lr $learning_rate -n_epochs $n_epochs --network $network --out_file ${results_prefix}${epsilon}_${dim}_${s_m} --sampling_method $s_m --width_problem $width_problem --stop_iter $stop_iter --var_dim $dim --epsilon $epsilon 
		done
	done
done