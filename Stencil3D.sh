#!/bin/bash -x
#
# RUN: qsub -g tgh-17IAC RDMA.sh 
#
# 資源タイプ=個数
#$ -l f_node=1
# 経過時間
#$ -l h_rt=0:20:00
#
# カレントディレクトリでジョブを実行する指定
#$ -cwd
# -e n2n.err
# -o n2n.out
#$ -m abe
#$ -M us152025@cc.seikei.ac.jp

. /etc/profile.d/modules.sh

# 2017
#module load intel-mpi/17.3.196 

module purge
module load cuda openmpi
module list

cat ${PE_HOSTFILE} | awk '{print $1}' > ${TMPDIR}/machines
echo "${TMPDIR}/machines"

pwd

for size in 128 512 1024
do
	for step in 128 1024
	do
		./Binary/Stencil_3D1GPU_${size}_${step}
		./Binary/Stencil_3DMultiGPU_${size}_${step}
		./Binary/Stencil_P2P_${size}_${step}
	done
done
cat $PE_HOSTFILE
date



echo "finish"
