#!/bin/bash
#SBATCH --job-name=gitlab-ci-dagger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-test-out.txt
#SBATCH --error=slurm-test-err.txt
#SBATCH --exclusive

GIT_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo Started: $(date)
echo Host: $(hostname)
echo Path: $GIT_ROOT
echo --------------------------------------------------------------------------------

# Setup Environment
cd $GIT_ROOT
unset MODULEPATH_ROOT
unset MODULEPATH
source /etc/profile.d/lmod.sh
source $CONDA_ROOT/etc/profile.d/conda.sh
source /auto/software/iris/setup_system.source
source $IRIS_INSTALL_ROOT/setup.source
# source /auto/ciscratch/conda/etc/profile.d/conda.sh
pushd apps/dagger
conda env create --force -p ./envs -f dagger.yaml
conda activate ./envs

# Run command
./run-policy-evaluation.sh
popd

# Collect Output
grep -e '[E]' slurm-test-out.txt > errors.txt

echo --------------------------------------------------------------------------------
echo Finished: $(date)
echo Errors: $(wc -l errors.txt)