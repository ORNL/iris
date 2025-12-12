#!/usr/bin/env python3

import iris
import pdb
import os
import numpy as np

# Use qiskit (pip install qiskit)
import qiskit
from qiskit import QuantumCircuit

# Adding the transpiler to reduce the circuit to QASM instructions
# supported by the backend
from qiskit import transpile 

# Use AerSimulator (pip install qiskit-aer)
from qiskit_aer import AerSimulator


os.environ["IRIS_ARCHS"] = "openmp:qiree"
iris.init()

def qiskit_task(spec, param1, param2, output):
    # Create a Quantum Circuit
    circ = QuantumCircuit(3)
    # Add a H gate on qubit 0, putting this qubit in superposition.
    circ.h(0)
    # Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
    # the qubits in a Bell state.
    circ.cx(0, 1)
    # Add a CX (CNOT) gate on control qubit 0 and target qubit 2, putting
    # the qubits in a GHZ state.
    circ.cx(0, 2)

    # Map the quantum measurement to the classical bits
    meas = QuantumCircuit(3, 3)
    meas.measure(range(3), range(3))
    
    # The Qiskit circuit object supports composition.
    # Here the meas has to be first and front=True (putting it before) 
    # as compose must put a smaller circuit into a larger one.
    qc = meas.compose(circ, range(3), front=True)

    backend = AerSimulator()
    qc_compiled = transpile(qc, backend)
    job_sim = backend.run(qc_compiled, shots=1024)
    result_sim = job_sim.result()

    counts = result_sim.get_counts(qc_compiled)
    iris.Quantum.qiskit2qiris(counts, output)
    print("\n Total Qiskit counts for 000 and 111 are:", counts)
    print("\n Total QIRIS counts for 000 and 111 are:", output)
    circ.draw(filename = 'circ-qiskit.txt')
    
    return iris.iris_ok

param1 = iris.dmem("Hello")
param2 = iris.dmem("World")
output = iris.dmem(np.empty(100, dtype=iris.CQireeResultRecord))

qtask = iris.task(qiskit_task, 1, [], [1], [], 
        [   (param1, iris.iris_r),
            (param2, iris.iris_r),
            (output, iris.iris_w) ]
)
qtask.submit(iris.iris_cpu)

iris.finalize()
