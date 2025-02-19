#!/usr/bin/env python3

import iris
import pdb
import os

# Use qiskit (pip install qiskit)
import qiskit
from qiskit import QuantumCircuit

# Use AerSimulator (pip install qiskit-aer)
from qiskit_aer import AerSimulator

home = os.getenv("HOME")

def go_quantum(iris_params, iris_dev):

    params = iris.iris2py(iris_params)
    dev = iris.iris2py_dev(iris_dev)
 
    circ_qasm = params[0]
    exp_num = params[1]
    
    with open('qcut_ghz_results.txt', 'a') as file:
        file.write('{\n')
    
    for i in range(3):
        circ = circ_qasm(exp_num,i)
        meas = QuantumCircuit(2, 2)
        meas.measure(range(2), range(2))
        qc = meas.compose(circ, range(2), front=True)
        backend = AerSimulator()
        job_sim = backend.run(qc, shots=1024)
        result_sim = job_sim.result()
        counts = result_sim.get_counts(qc)
        if i==0 or i==1:
            with open('qcut_ghz_results.txt', 'a') as file:
                file.write(str(counts)+',\n')
        else:
            with open('qcut_ghz_results.txt', 'a') as file:
                file.write(str(counts)+'\n')
        print(counts)
    
    with open('qcut_ghz_results.txt', 'a') as file:
        file.write('},\n')
        
    return iris.iris_ok

iris.init()

n = 64

tasks = [iris.task() for i in range(n)]

pool = []

for index in range(n):
    # these circuits are written in qasm and will not need transpiling
    # TODO: convert circuits to QIR for qiree processing
    path = home+"/qiree/examples/others/qcut_circuits_ghz/"
    circ_qasm = lambda i, j: QuantumCircuit.from_qasm_file(path+'circ_'+f"{i:02}"+'_'+str(j)+'.txt')
    parameters = [circ_qasm, index]
    tasks[index].pyhost(go_quantum, parameters)
    tasks[index].submit(iris.iris_default, sync=0)
    pool.append(parameters)

iris.synchronize()
iris.finalize()
