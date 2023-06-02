.. index:: ! hello world

Hello World
==================

Setting the following environment variables is highly recommended to make life easier.

.. code-block:: bash

  $ source <install_path>/setup.source # default install_path would be $HOME/.iris


The “Hello World” program is the first step towards learning IRIS. This program displays the message “HELLO WORLD” on the screen.

.. code-block:: bash

  $ cd iris/apps/helloworld
  $ make
  $ ./helloworld
  HELLO WORLD
  $

Host Code
---------

.. content-tabs::

    .. tab-container:: tab1
        :title: C

        .. literalinclude:: ../../../apps/helloworld/helloworld.c
          :language: c

    .. tab-container:: tab2
        :title: C++

        .. literalinclude:: ../../../apps/helloworld/helloworld.cpp
          :language: cpp

Kernels
-------

.. content-tabs::

    .. tab-container:: tab1
        :title: CUDA

        .. literalinclude:: ../../../apps/helloworld/kernel.cu
          :language: c

    .. tab-container:: tab2
        :title: HIP

        .. literalinclude:: ../../../apps/helloworld/kernel.hip.cpp
          :language: c

    .. tab-container:: tab3
        :title: OpenCL

        .. literalinclude:: ../../../apps/helloworld/kernel.cl
          :language: c

    .. tab-container:: tab4
        :title: OpenMP

        .. literalinclude:: ../../../apps/helloworld/kernel.openmp.h
          :language: c

.. index:: ! saxpy

SAXPY
=====

SAXPY stands for "Single-precision A * X Plus Y". It is a combination of scalar multiplication and vector addition.

.. code-block:: bash

  $ cd iris/apps/saxpy
  $ make
  $ ./saxpy-c
  X [  0.  1.  2.  3.  4.  5.  6.  7.]
  Y [  0.  1.  2.  3.  4.  5.  6.  7.]
  S = 10.000000 * X + Y [   0.  11.  22.  33.  44.  55.  66.  77.]
  $

Host Code
---------

.. content-tabs::

    .. tab-container:: tab1
        :title: C

        .. literalinclude:: ../../../apps/saxpy/saxpy.c
          :language: c

    .. tab-container:: tab2
        :title: C++

        .. literalinclude:: ../../../apps/saxpy/saxpy.cpp
          :language: cpp

    .. tab-container:: tab3
        :title: Fortran

        .. literalinclude:: ../../../apps/saxpy/saxpy.f90
          :language: fortran

    .. tab-container:: tab4
        :title: Python

        .. literalinclude:: ../../../apps/saxpy/saxpy.py
          :language: python

Kernels
-------

.. content-tabs::

    .. tab-container:: tab1
        :title: CUDA

        .. literalinclude:: ../../../apps/saxpy/kernel.cu
          :language: c

    .. tab-container:: tab2
        :title: HIP

        .. literalinclude:: ../../../apps/saxpy/kernel.hip.cpp
          :language: c

    .. tab-container:: tab3
        :title: OpenCL

        .. literalinclude:: ../../../apps/saxpy/kernel.cl
          :language: c

    .. tab-container:: tab4
        :title: OpenMP

        .. literalinclude:: ../../../apps/saxpy/kernel.openmp.h
          :language: c

.. index:: ! data memory

Data Memory
==================

One of the major benefits of using IRIS is its "data memory" feature, which automatically manage data movement independent of the device scheduling. Here is an example of the use of data memory during a vector addition code. Note how the:

.. code-block:: c++

  iris_data_mem_create(&mem_A, A, SIZE * sizeof(int));
  ...
  iris_task_dmem_flush_out(task0,mem_C);

call differs from the SAXPY example above. We no longer need ``iris_task_h2d_full`` and ``iris_task_d2h_full`` calls, instead, we only need to know when to flush the final memory transfer required by the host. This is a simpler work-flow that the conventional explicit memory movement approach.

Running
-------

.. code-block:: bash

  $ cd iris/apps/vecadd
  $ make
  $ ./vecadd-iris

Host Code
---------

.. content-tabs::

    .. tab-container:: tab1
        :title: C++

        .. literalinclude:: ../../../apps/vecadd/vecadd-iris.cpp
          :language: cpp


Kernels
-------

.. content-tabs::

    .. tab-container:: tab1
        :title: CUDA

        .. literalinclude:: ../../../apps/vecadd/kernel.cu
          :language: c

    .. tab-container:: tab2
        :title: HIP

        .. literalinclude:: ../../../apps/vecadd/kernel.hip.cpp
          :language: c

    .. tab-container:: tab3
        :title: OpenCL

        .. literalinclude:: ../../../apps/vecadd/kernel.cl
          :language: c

    .. tab-container:: tab4
        :title: OpenMP

        .. literalinclude:: ../../../apps/vecadd/kernel.openmp.h
          :language: c


.. index:: ! device selection


Device Selection
==================

IRIS opportunistically attempts to use all available devices and backends, it resolves task names to function names in the corresponding kernel binaries.
It allows device selection to be set both at compile and at runtime.

Compile Time
-------------

The user can submit the device target(s) for when the task is submitted:

.. code-block:: c++

  iris_task_submit(iris_task task, int device, const char* opt, int sync);

This task submission includes information about the task, such as a hint, target device parameter, synchronization mode (blocking or non-blocking), and policy selector that indicates where the task should be executed.
The ``device`` is the *device submission policy*. The complete list of available targets are:

======================== ======================================================================
Device Policy            About
======================== ======================================================================
iris_cpu                 Submit the task to a CPU device
iris_gpu                 Submit the task to any GPU device
iris_fpga                Submit the task to any FPGA  (currently Intel and Xilinx)
iris_dsp                 Submit the task to any DSP device (currently Hexagon)
iris_nvidia              Submit the task to an NVIDIA GPU device
iris_amd                 Submit the task to an AMD GPU device
iris_gpu_intel           Submit the task to an Intel GPU device
iris_phi                 Submit the task to an Intel Xeon Phi device
======================== ======================================================================

We can also submit tasks according to a *scheduling policy*:

======================== =============================================================================================================================================
Scheduling Policy        About
======================== =============================================================================================================================================
iris_default             Use the first device
iris_roundrobin          Submit this task in a round-robin (cyclic) way, for equal work sharing
iris_depend              Submit this task to a device that has been assigned its dependent
iris_data                Submit task to device to minimize data movement
iris_profile             Submit the task to the device based on execution time history
iris_random              Randomly assign this task to any of the available devices
iris_pending             Delay submitting the task until the memory it depends on has been assigned, then use that device
iris_any                 Submit task to the device with the fewest assigned tasks
iris_all                 Submit the task to all device queues, the device that accesses it first has exclusive execution (it is removed from the other device queues)
iris_custom              Submit the task based on a used provided, custom policy
======================== =============================================================================================================================================

The ``opt`` parameter is for ``iris_custom`` policies.

Runtime
-------------

We can also filter out devices at runtime by setting the ``IRIS_ARCHS`` environment variable. Modifying the selection of backends to instantiate allows dynamic device targets---without requiring recompilation. All current options are:``hip``,``cuda``,``opencl``, and ``openmp``. An example of only allowing execution on ``openmp`` and ``hip`` devices would then be:

.. code-block:: bash

  $ export IRIS_ARCHS=hip,openmp
  $ ./helloworld
  HELLO WORLD
  $


