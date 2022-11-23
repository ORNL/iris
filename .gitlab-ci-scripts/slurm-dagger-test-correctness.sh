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

# Default conda environment variable.
if [ ! -n "$CONDA_ROOT" ]; then
	CONDA_ROOT="/noback/$USER/conda"
fi

echo Started: $(date)
echo Host: $(hostname)
echo Path: $GIT_ROOT
echo --------------------------------------------------------------------------------

cd $GIT_ROOT

### Setup Environment

## Slurm/gitlab-runner specific setup
# Work around for lmod + Slurm
unset MODULEPATH_ROOT
unset MODULEPATH
source /etc/profile.d/lmod.sh
# Load conda
source $CONDA_ROOT/etc/profile.d/conda.sh

# General setup for IRIS
source /auto/software/iris/setup_system.source
source $IRIS_INSTALL_ROOT/setup.source

# Local conda environment setup
pushd apps/dagger
conda env create --force -p ./envs -f dagger.yaml
conda activate ./envs

env

# Run command
./test_correctness.sh
popd

# Collect Output
grep -e '\[E\]' slurm-test-out.txt > errors.txt

echo --------------------------------------------------------------------------------
echo Finished: $(date)
echo Errors: $(wc -l errors.txt)