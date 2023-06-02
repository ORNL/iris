Build from Source
==================

IRIS uses `CMake (>= 2.8) <https://cmake.org>`_ for building, testing, and installing the library.

.. code-block:: bash
   
  $ git clone git@github.com:ornl/iris.git
  $ cd iris
  $ mkdir build
  $ cd build
  $ cmake .. -DCMAKE_INSTALL_PREFIX=<install_path> # $HOME/.local is good for install_path.
  $ make -j
  $ ctest # Running tests is optional.
  $ make install

