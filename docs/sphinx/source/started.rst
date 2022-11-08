.. index:: ! hello world

Hello World
==================

Setting the following environment variables is highly recommended to make life easier.

.. code-block:: bash

  $ export IRIS=<install_path> # install_path would be $HOME/.local
  $ export CPATH=$CPATH:$IRIS/include
  $ export LIBRARY_PATH=$LIBRARY_PATH:$IRIS/lib:$IRIS/lib64
  $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$IRIS/lib:$IRIS/lib64
  $ export PYTHONPATH=$PYTHONPATH:$IRIS/include
 

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

        .. literalinclude:: _code/helloworld/helloworld.c
          :language: c

    .. tab-container:: tab2
        :title: C++

        .. literalinclude:: _code/helloworld/helloworld.cpp
          :language: cpp

Kernels
-------

.. content-tabs::

    .. tab-container:: tab1
        :title: CUDA

        .. literalinclude:: _code/helloworld/kernel.cu
          :language: c

    .. tab-container:: tab2
        :title: HIP

        .. literalinclude:: _code/helloworld/kernel.hip.cpp
          :language: c

    .. tab-container:: tab3
        :title: OpenCL

        .. literalinclude:: _code/helloworld/kernel.cl
          :language: c

    .. tab-container:: tab4
        :title: OpenMP

        .. literalinclude:: _code/helloworld/kernel.omp.h
          :language: c

    .. tab-container:: tab5
        :title: Hexagon

        .. literalinclude:: _code/helloworld/kernel.hexagon.c
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

        .. literalinclude:: _code/saxpy/saxpy.c
          :language: c

    .. tab-container:: tab2
        :title: C++

        .. literalinclude:: _code/saxpy/saxpy.cpp
          :language: cpp

    .. tab-container:: tab3
        :title: Fortran

        .. literalinclude:: _code/saxpy/saxpy.f90
          :language: fortran

    .. tab-container:: tab4
        :title: Python

        .. literalinclude:: _code/saxpy/saxpy.py
          :language: python

Kernels
-------

.. content-tabs::

    .. tab-container:: tab1
        :title: CUDA

        .. literalinclude:: _code/saxpy/kernel.cu
          :language: c

    .. tab-container:: tab2
        :title: HIP

        .. literalinclude:: _code/saxpy/kernel.hip.cpp
          :language: c

    .. tab-container:: tab3
        :title: OpenCL

        .. literalinclude:: _code/saxpy/kernel.cl
          :language: c

    .. tab-container:: tab4
        :title: OpenMP

        .. literalinclude:: _code/saxpy/kernel.omp.h
          :language: c

    .. tab-container:: tab5
        :title: Hexagon

        .. literalinclude:: _code/saxpy/kernel.hexagon.c
          :language: c

