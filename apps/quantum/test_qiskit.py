#!/usr/bin/env python3

import iris
import pdb
import qiskit
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit import IBMQ
IBMQ.save_account('#')

iris.init()
#from qiskit.visualization import *
provider = qiskit.IBMQ.load_account()


def go_quantum(iris_params, iris_dev):
    params = iris.iris2py(iris_params)
    dev = iris.iris2py_dev(iris_dev)
    print(params)

    qiskit.IBMQ.save_account('#')
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    simulator = qiskit.Aer.get_backend('qasm_simulator')
    job = qiskit.execute(circuit, simulator, shots = 1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print("\n Total count for 00 and 11 are:", counts)
    circuit.draw(filename = 'fig-qiskit.txt')
    plot_histogram(counts)

    return iris.iris_ok

task1 = iris.task()
task1.pyhost(go_quantum, [123])
task1.submit(iris.iris_cpu)

iris.finalize()
