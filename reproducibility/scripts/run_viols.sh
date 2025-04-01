#!/bin/bash
#SBATCH -J fxcrt_interval_length_expt  # A single job name for the array
#SBATCH -p shared,seas_compute,janson_cascade,janson# Partition
#SBATCH -c 1 # number of cores
#SBATCH -t 1-00:00  # Running time in the format - D-HH:MM
#SBATCH --mem 4000 # Memory request - 1000 corresponds to 1GB
#SBATCH -o out_%a.out # Standard output
#SBATCH -e err_%a.err # Standard error
python violations.py # this indicates which file to run
#SBATCH --mail-type=END #Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=dpaulson@college.harvard.edu #Email to which notifications will be sent
