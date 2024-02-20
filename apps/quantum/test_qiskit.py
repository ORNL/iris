#!/usr/bin/env python3

import iris
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit import IBMQ
IBMQ.save_account('#')

iris.init()

provider = IBMQ.load_account()

def go_quantum(params, dev):
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots = 1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print("\n Total count for 00 and 11 are:", counts)
    circuit.draw(filename = 'fig-qiskit.txt')
    plot_histogram(counts)

    return iris.iris_ok

task1 = iris.task()
task1.host(go_quantum, [])
task1.submit(iris.iris_cpu)

iris.finalize()
