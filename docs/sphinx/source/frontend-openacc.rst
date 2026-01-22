Build OpenACC/OpenMP Frontend
=============================

IRIS uses `OpenARC <https://github.com/ORNL/OpenARC>`_ as its OpenACC/OpenMP frontend compiler.
This frontend is needed only when compiling OpenACC/OpenMP programs into IRIS.

Prerequisites
------------

1. JAVA SE 7 or later
2. GCC 4.2 or later
3. IRIS 3.0 or later

Environment Variables Setup
---------------------------

.. code-block:: bash

  $ export OPENARC_ARCH=6 #Set OpenARC target architecture to IRIS
  $ export openarc=<openarc_repository_root_directory>
  $ export OPENARC_INSTALL_ROOT=<openarc_install_root_directory>

Build OpenARC 
-------------

.. code-block:: bash
   
  $ git clone git@github.com:ORNL/OpenARC.git
  $ cd OpenARC
  $ mkdir build
  $ cd build
  $ cmake .. -DCMAKE_INSTALL_PREFIX=$OPENARC_INSTALL_ROOT \ # If not specified, $openarc/install is used.
  $          -DCMAKE_CXX_COMPILER=g++
  $ make -j
  $ make install

To run the tests

.. code-block:: bash
   
  $ cd $openarc/test/bin
  $ batchTest.bash

Optional environment variables used by the OpenARC-generated IRIS program:

.. code-block:: bash
   
  $ export OPENARCRT_IRIS_DMEM=1 # Use DMEM (default)
                               0 # Do not use DMEM (use IRIS MEM) 
  $ export OPENARCRT_IRIS_POLICY=<iris_scheduling_policy>
           # <iris_scheduling_policy> can be any IRIS policy (e.g., iris_roundrobin) or none.
           # If none, do not use any IRIS policy and use a target device specified by the OpenACC/OpenMP environment.
           
