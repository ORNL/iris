#!/usr/bin/env python3

import iris
import pdb

# Use qiskit (pip install qiskit)
import qiskit
from qiskit import QuantumCircuit
import copy


# Adding the transpiler to reduce the circuit to QASM instructions
# supported by the backend
from qiskit import transpile 
from multiprocessing import Process


# Use AerSimulator (pip install qiskit-aer)
from qiskit_aer import AerSimulator

def go_run_quantum_core(params):

    # print(params)

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
    print(f"Total {params[0]} counts for 000 and 111 are:", counts)
    return iris.iris_ok

def go_run_quantum(iris_params, iris_dev):
    params = iris.iris2py(iris_params)
    dev = iris.iris2py_dev(iris_dev)
    p1 = Process(target=go_run_quantum_core, args=(params,))
    p1.start()
    p1.join()
    return iris.iris_ok

iris.init()

# Running in parallel
n = 64

# Create n tasks (no communications with each other)
tasks = [iris.task() for i in range(n)]

for i in range(n):
    parameters = [i]
    tasks[i].pyhost(go_run_quantum, parameters)
    tasks[i].submit(iris.iris_default, sync=0)
print("Exited submissions!")
iris.synchronize()
print("Completed synchronize of ntasks:", len(tasks))
iris.finalize()
print("IRIS finalized!")
