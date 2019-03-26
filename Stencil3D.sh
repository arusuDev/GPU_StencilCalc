#!/bin/bash -x
#
# RUN: qsub -g tgh-17IAC RDMA.sh 
#
# 資源タイプ=個数
#$ -l f_node=1
# 経過時間
#$ -l h_rt=0:02:00
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

cat $PE_HOSTFILE
date
./Stencil_GPU/Stencil_3D1GPU
./Stencil_GPU/Stencil_3DMultiGPU
./Stencil_P2P
echo "finish"
