#!/bin/bash
#SBATCH --job-name=test-correctness-gitlab-ci-dagger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-test-out.txt
#SBATCH --error=slurm-test-err.txt
#SBATCH --exclusive
#SBATCH --time="1:01:00"

GIT_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Default conda environment variable.
#if [ ! -n "$CONDA_ROOT" ]; then
#	CONDA_ROOT="/noback/$USER/conda"
#fi

cd $GIT_ROOT

echo Started: $(date)
echo Host: $(hostname)
echo Path: $GIT_ROOT
echo Groups: $(groups)
echo ${SLURM_JOB_ID} > slurm.job
echo --------------------------------------------------------------------------------

### Setup Environment

## Slurm/gitlab-runner specific setup
# Work around for lmod + Slurm
unset MODULEPATH_ROOT
unset MODULEPATH
source /etc/profile.d/lmod.sh
module load cmake
module load gnu

# Load conda
#source $CONDA_ROOT/etc/profile.d/conda.sh

# General setup for IRIS
source /auto/software/iris/setup_system.source
IRIS_TAG=.$IRIS_TAG.$IRIS_MACHINE.$IRIS_TESTNAME IRIS_INSTALL_ROOT=$IRIS_INSTALL_ROOT bash build.sh -DENABLE_FFI=$IRIS_FFI_FLAG -DCMAKE_BUILD_TYPE=DEBUG -DCOVERAGE=true 
source $IRIS_INSTALL_ROOT.$IRIS_TAG.$IRIS_MACHINE.$IRIS_TESTNAME/setup.source

set -e

# Local conda environment setup
if [ "x$IRIS_TAG" = "x" ]; then
echo "Working on apps/dagger"
else
cp -r apps/dagger apps/dagger$IRIS_TAG
echo "Working on apps/dagger$IRIS_TAG"
fi
pushd apps/dagger$IRIS_TAG
#conda env create --force -p ./envs -f dagger.yaml
#conda activate ./envs

# Run command
make -f Makefile.venv clean
make -f Makefile.venv setup
make -f Makefile.venv test-correctness
make -f Makefile.venv validate-test-correctness
#./test_correctness.sh
popd

# Collect Output
#grep -e '\[E\]' slurm-test-out.txt > errors.txt
#grep -e 'EnvironmentLocationNotFound' slurm-test-err.txt >> errors.txt

# Validate Json
#pushd apps/dagger
#for f in *-graph.json
#do
#   echo Validating Json Schema for file $f.
#   $GIT_ROOT/utils/validate_schema.py -i $f -s $GIT_ROOT/schema/dagger.schema.json || echo Json Validation failed for $f >> errors.txt
#done
#popd

#echo --------------------------------------------------------------------------------
#echo Finished: $(date)
#echo Errors: $(wc -l errors.txt)
