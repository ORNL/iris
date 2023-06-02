Build from Source
==================

IRIS uses `CMake (>= 2.8) <https://cmake.org>`_ for building, testing, and installing the library.

.. code-block:: bash
   
  $ git clone git@github.com:ornl/iris.git
  $ cd iris
  $ mkdir build
  $ cd build
  $ cmake .. -DCMAKE_INSTALL_PREFIX=<install_path> # $HOME/.iris is good for the install_path.
  $ make -j
  $ make install

To run the tests

.. code-block:: bash
   
  $ cd ../tests
  $ mkdir build
  $ cd build
  $ cmake ..
  $ make -j
  $ make test

