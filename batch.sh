#!/bin/bash -l
#NOTE the -l flag!
# myFirstScript.sh used to showcase the basic slurm
# commands. Modify this to suit your needs.
# Name of the job -You'll probably want to customize this
#SBATCH -J Citizenly-RL-SnowPlowing
# Use the resources available on this account
#SBATCH -A citizenly
#Standard out and Standard Error output files 
#SBATCH -o out-04-06-20-1600.o
#SBATCH -e err-04-06-20-1600.o
#To send mail for updates on the job
#SBATCH --mail-user=mf3791@rit.edu
#SBATCH --mail-type=ALL
#Request 4 Days, 6 Hours, 5 Minutes, 3 Seconds run time MAX, 
# anything over will be KILLED
#SBATCH -t 0-12:00:00
# Put in debug partition for testing small jobs, like this one
# But because our requested time is over 1 day, it won't run, so# use any tier you have available
#SBATCH -p tier3
# Request 4 cores for one task, note how you can put multiple commands
# on one line
#SBATCH -n 1 -c 2
#Job memory requirements in MB
#SBATCH --mem=20G
#GPU USAGE
#SBATCH --gres=gpu:p4:1

#Module loading
#spack load py-horovod
# spack load py-torchvision@0.4.0 ^py-numpy@1.16.2
# spack load py-matplotlib ^python@3.6.8 ^py-numpy@1.16.2 ^sqlite@3.26.0~column_metadata~fts3~fts5~functions~rtree
# spack load opencv ^python@3.6.8 ^py-numpy@1.16.2
spack env activate ml-geo-20070801
#Job script goes below this line
#
echo " (${HOSTNAME}) Job Running..."
python3 trainer.py --task=vrp --nodes=50
echo " *(${HOSTNAME}) Job completed. "