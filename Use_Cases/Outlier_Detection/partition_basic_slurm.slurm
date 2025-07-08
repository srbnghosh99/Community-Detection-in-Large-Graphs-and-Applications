#!/bin/bash

#SBATCH --partition=Orion
#SBATCH --job-name=ciao_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=5:00:00
#SBATCH --array=0-25
source /users/sghosh15/ga_env/bin/activate 


TOTAL=7419
#Total=144726
# Constants
SOLUTIONS_PER_JOB=300

# Calculate start and end index for this job
START=$(( SLURM_ARRAY_TASK_ID * SOLUTIONS_PER_JOB ))
END=$(( START + SOLUTIONS_PER_JOB ))

if [ $END -gt $TOTAL ]; then
    END=$TOTAL
fi

echo "Running job $TASK_ID: processing solutions from $START to $END"

cd ..
# Run your processing script with the current chunk range
./yelp.sh $START $END


