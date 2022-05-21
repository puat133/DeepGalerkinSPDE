#!/usr/bin/bash

optimizer=${1:-hf}
prior=${2:-1}
kappa0=5e2
nsteps=10000
meas_samples=100
learn=1e-5
softmax="softmax"
smallerlearn=$learn

if [ "$optimizer" == "hf" ];then
	kappa0=1e0
	nsteps=100
	learn=1e-1
	smallerlearn=1e-2
	softmax="nosoftmax"
elif [ "$optimizer" == "adahessian" ];then
	learn=1e-2
	smallerlearn=$learn
elif [ "$optimizer" == "adam" ]; then
  learn=1e-4
  smallerlearn=$learn
fi



sbatch run_one_d_sim_cpu_xxxxx.slrm $optimizer SMOOTH $softmax $kappa0 $smallerlearn $meas_samples $nsteps cuda $prior
sbatch run_one_d_sim_cpu_xxxxx.slrm $optimizer RECT $softmax $kappa0 $learn $meas_samples $nsteps cuda $prior
# sbatch run_one_d_sim_cpu_xxxxx.slrm $optimizer BOX $softmax $kappa0 $learn $meas_samples $nsteps cuda
# sbatch run_one_d_sim_cpu_xxxxx.slrm $optimizer RECT2 $softmax $kappa0 $learn $meas_samples $nsteps cuda

