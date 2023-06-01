    DAGGER: Directed Acyclic Graph Generator for Evaluating Runtimes
------------------------------------------------------------------------

Dagger is a simple python program `./dagger_generator.py` which generates task-graphs of arbitary length and complexity.

##Installation

The generator and visualization tools use python. All packages are installed via the conda package manager.
To create a working environment, it should be as simple as:

```
  conda env create -f dagger.yaml
  conda activate dagger
```

**Note** the DAGGER visualization uses a custom version of networkx which can be installed with `python -m pip install "networkx @ git+https://github.com/BeauJoh/networkx.git@main"`.

## Usage:

```
DAGGER: Directed Acyclic Graph Generator for Evaluating Runtimes

Arguments:
  -h, --help                        show this help message and exit
      --kernels KERNELS             The kernel names --in the current directory-- to generate tasks,
                                      presented as a comma separated value string e.g. "process,matmul"
      --kernel-split KERNEL_SPLIT   The percentage of each kernel being assigned to the task,
                                      presented as a comma separated value string e.g. "80,20".
      --depth DEPTH                 Depth of tree, e.g. 10.
      --num-tasks NUM_TASKS         Total number of tasks to build in the DAG, e.g. 100.
      --min-width MIN_WIDTH         Minimum width of the DAG, e.g. 1.
      --max-width MAX_WIDTH         Maximum width of the DAG, e.g. 10.
      --cdf-mean CDF_MEAN           Mu of the Cumulative Distribution Function, default=0
      --cdf-std-dev CDF_STD_DEV     Sigma^2 of the Cumulative Distribution Function, default=0.2
      --duplicates                  The # of times to reproduce the DAG horizontally across the tree (increasing complexity)
      --buffers-per-kernel          Key value pairs with the kernel name and the type of memory required
      --kernel-dimensons            Key value pairs to assign the dimensionality of each kernel
      --skips SKIPS                 Maximum number of jumps down the DAG levels (Delta) between tasks, default=1
```

## Sample Usage:
To generate a linear/sequential series of tasks---10 tasks long (excluding start and stop nodes), with the sole kernel function called `bigk`:
```
./dagger_generator.py --kernels="bigk" --duplicates="0" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1
```
The kernel-split argument specify the probability (as a percentage) of the corresponding kernel name being used.
Default outputs are the DAG as a JSON file `graph.json`, and a visualization of the DAG `dag.png`.

##Evaluating IRIS' Naïve Scheduling Policies
There is a provided Dagger **runner** which loads the generated output DAG (`.json`) and creates a task-graph which is then submitted to IRIS for execution.
This allows us to evaluate potential scheduling policies on arbitarily complex workloads with hairy dependencies.
It automatically creates and assigns the required number of memory buffers.
Fortunately, it also accepts (largely) the same input arguments as were provided to generate the graph with the DAGGER generator.
The only difference is it has another 3 *required* arguments:

```
--repeats             An integer to indicate the number of times to resubmit the task-graph (for statistical rigor)
--scheduling-policy   A string stating which of IRIS' scheduling policies to use
--size                Integer specifying the size of the payload---the number of elements used in each buffer
```

###Running

The command to run our previously generated DAG:

```
IRIS_HISTORY=1 ./dagger_runner --repeats=1 --scheduling-policy=any --size=1024  --kernels="bigk" --duplicates="0" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="bigk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 ; mv dagger_runner-$SYSTEM-*.csv linear-10-any.csv
```
Setting the `IRIS_HISTORY=1` environment variable enables IRIS to generate an execution trace (of the format `$APPNAME-$SYSTEM-$TIMESTAMP`.csv)

###Plotting
To plot this result:

  ./plotter.py ./linear-10-any.csv linear-10-any.pdf "Linear 10 dataset with ANY scheduling policy" "Init"

###Reproducibility Artifacts
All results to evaluate IRIS' scheduling policies with some interesting DAGGER inputs is provided in a single executable script:
  `./run-policy-evaluation.sh`

