cmake_minimum_required(VERSION 3.5)

project(IRIS)
include(GNUInstallDirs)

set(IRIS_VERSION_MAJOR 1)
set(IRIS_VERSION_MINOR 0)
set(IRIS_VERSION_PATCH 0)

set(IRIS_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(IRIS_INC_DIR ${PROJECT_SOURCE_DIR}/include)
set(IRIS_BIN_DIR ${PROJECT_BINARY_DIR}/src)

option(DEBUG_DISABLE    "Disable debug messages"    ON)
option(TRACE_DISABLE    "Disable trace messages"    ON)
option(TODO_DISABLE     "Disable todo messages"     ON)
option(INFO_DISABLE     "Disable info messages"     ON)
option(CHECK_DISABLE    "Disable checking messages" ON)
option(WARNING_DISABLE  "Disable warning messages"  ON)
option(DISABLE_DYNAMIC_LINKING "Disable dynamic linking"     OFF )
option(DISABLE_CONSISTENCY_CHECK "Disable memry consistency checks "     OFF )
option(ENABLE_SINGLE_DEVICE_PER_CU "Enable single device per compute unit "     OFF )
option(USE_PYTHON       "Enable support for Python"     ON)
option(USE_RISCV        "Enable support for RISCV"      OFF)
option(USE_FORTRAN      "Enable support for Fortran 90" OFF)

option(USE_SIGHANDLER   "Enable signal handler"         OFF)
option(USE_HUB          "Enable hub"                    OFF)
option(ENABLE_FFI       "Enable FFI based kernel calls" OFF)
option(IRIS_ASYNC_STREAMING "Enable IRIS CUDA/HIP Asynchronous streaming"                    OFF)

if (USE_RISCV)
set(CMAKE_C_COMPILER   riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)
set(DISABLE_DYNAMIC_LINKING  ON)
endif() # USE_RISCV

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g") 
set(CMAKE_C_FLAGS_RELEASE "-O3 -g")   

macro ( info NAME )
  string(ASCII 27 Esc)
  message ( "${Esc}[35m${NAME}${Esc}[m" )
endmacro ()
if(NOT ${DEBUG_DISABLE})
  info("Building with DEBUG logging.")
endif()
if(NOT ${TRACE_DISABLE})
  info("Building with TRACE logging.")
endif()
if(NOT ${TODO_DISABLE})
  info("Building with TODO logging.")
endif()
if(NOT ${INFO_DISABLE})
  info("Building with INFO logging.")
endif()
if(NOT ${CHECK_DISABLE})
  info("Building with CHECK logging.")
endif()
if(NOT ${WARNING_DISABLE})
  info("Building with WARNING messages.")
endif()

add_subdirectory(include)
add_subdirectory(src)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/utils DESTINATION ${CMAKE_INSTALL_PREFIX}/.)

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/setup.source
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMAND echo "SCRIPT_FILE=\${BASH_SOURCE[0]}" > ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "PYTHON_EXE=python" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "version=\$(python -V 2>&1 | grep -Po '(?<=Python )(.+)' || echo '')" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "if [[ -z \"\$version\" ]]" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "then" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "    PYTHON_EXE=python3" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "    version=\$(python3 -V 2>&1 | grep -Po '(?<=Python )(.+)' || echo '')" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "    if [[ -z \"\$version\" ]]" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "    then" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "    echo \"No Python is detected! Cannot source\" " >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "    return" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "    fi" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "fi" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "SCRIPT=$(\$PYTHON_EXE -c \"import os,sys; print(os.path.realpath(os.path.expanduser(sys.argv[1])))\" \${SCRIPT_FILE})" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "export IRIS=$(dirname -- \${SCRIPT})" >> ${CMAKE_BINARY_DIR}/setup.source
  COMMAND echo "export LD_LIBRARY_PATH=.:\$IRIS/lib:\$IRIS/lib64:\$LD_LIBRARY_PATH" >> ${CMAKE_BINARY_DIR}/setup.source
  COMMAND echo "export CPATH=\$IRIS/include:\$CPATH" >> ${CMAKE_BINARY_DIR}/setup.source
  COMMAND echo "export PYTHONPATH=\$IRIS:\$PYTHONPATH" >> ${CMAKE_BINARY_DIR}/setup.source
  COMMAND echo "export SYSTEM=\$(hostname|cut -d . -f 1|sed 's/[0-9]*//g')" >>  ${CMAKE_BINARY_DIR}/setup.source
  COMMAND echo "export MACHINE=\${SYSTEM^}" >>  ${CMAKE_BINARY_DIR}/setup.source
  COMMAND echo "echo \"*********************** IRIS Environent *********************\"" >> ${CMAKE_BINARY_DIR}/setup.source
  COMMAND echo "echo \"IRIS: \$IRIS\"" >> ${CMAKE_BINARY_DIR}/setup.source
  COMMAND echo "echo \"*************************************************************\"" >> ${CMAKE_BINARY_DIR}/setup.source
  )
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/setup.zsh
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMAND echo "SCRIPT_FILE=\${(%):-%N}" > ${CMAKE_BINARY_DIR}/setup.zsh VERBATIM
  COMMAND echo "PYTHON_EXE=python" >> ${CMAKE_BINARY_DIR}/setup.zsh VERBATIM
  COMMAND echo "version=\$(python -V 2>&1 | grep -Po '(?<=Python )(.+)' || echo '')" >> ${CMAKE_BINARY_DIR}/setup.zsh VERBATIM
  COMMAND echo "if [[ -z \"\$version\" ]]" >> ${CMAKE_BINARY_DIR}/setup.zsh VERBATIM
  COMMAND echo "then" >> ${CMAKE_BINARY_DIR}/setup.zsh VERBATIM
  COMMAND echo "    PYTHON_EXE=python3" >> ${CMAKE_BINARY_DIR}/setup.zsh VERBATIM
  COMMAND echo "    version=\$(python3 -V 2>&1 | grep -Po '(?<=Python )(.+)' || echo '')" >> ${CMAKE_BINARY_DIR}/setup.zsh VERBATIM
  COMMAND echo "    if [[ -z \"\$version\" ]]" >> ${CMAKE_BINARY_DIR}/setup.zsh VERBATIM
  COMMAND echo "    then" >> ${CMAKE_BINARY_DIR}/setup.zsh VERBATIM
  COMMAND echo "    echo \"No Python is detected! Cannot source\" " >> ${CMAKE_BINARY_DIR}/setup.zsh VERBATIM
  COMMAND echo "    return" >> ${CMAKE_BINARY_DIR}/setup.zsh VERBATIM
  COMMAND echo "    fi" >> ${CMAKE_BINARY_DIR}/setup.zsh VERBATIM
  COMMAND echo "fi" >> ${CMAKE_BINARY_DIR}/setup.zsh VERBATIM
  COMMAND echo "SCRIPT=$(\$PYTHON_EXE -c \"import os,sys; print(os.path.realpath(os.path.expanduser(sys.argv[1])))\" \${SCRIPT_FILE})" >> ${CMAKE_BINARY_DIR}/setup.zsh VERBATIM
  COMMAND echo "export IRIS=$(dirname -- \${SCRIPT})" >> ${CMAKE_BINARY_DIR}/setup.zsh
  COMMAND echo "export LD_LIBRARY_PATH=.:\$IRIS/lib:\$IRIS/lib64:\$LD_LIBRARY_PATH" >> ${CMAKE_BINARY_DIR}/setup.zsh
  COMMAND echo "export LIBRARY_PATH=.:\$IRIS/lib:\$IRIS/lib64:\$LIBRARY_PATH" >> ${CMAKE_BINARY_DIR}/setup.zsh
  COMMAND echo "export CPATH=\$IRIS/include:\$CPATH" >> ${CMAKE_BINARY_DIR}/setup.zsh
  COMMAND echo "export PYTHONPATH=\$IRIS:\$PYTHONPATH" >> ${CMAKE_BINARY_DIR}/setup.zsh
  COMMAND echo "export SYSTEM=\$(hostname|cut -d . -f 1|sed 's/[0-9]*//g')" >>  ${CMAKE_BINARY_DIR}/setup.zsh
  COMMAND echo "export MACHINE=\$(tr '[:lower:]' '[:upper:]' <<< \${SYSTEM:0:1})\${SYSTEM:1}" >>  ${CMAKE_BINARY_DIR}/setup.zsh
  COMMAND echo "echo \"*********************** IRIS Environent *********************\"" >> ${CMAKE_BINARY_DIR}/setup.zsh
  COMMAND echo "echo \"IRIS: \$IRIS\"" >> ${CMAKE_BINARY_DIR}/setup.zsh
  COMMAND echo "echo \"*************************************************************\"" >> ${CMAKE_BINARY_DIR}/setup.zsh
  )
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/setup.fish
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMAND echo "set SCRIPT_FILE (status filename)" > ${CMAKE_BINARY_DIR}/setup.fish VERBATIM
  COMMAND echo "set -l PYTHON_EXE python" >> ${CMAKE_BINARY_DIR}/setup.fish VERBATIM
  COMMAND echo "set -l pyversion (python -V 2>&1 | grep -Po '(?<=Python )(.+)' || echo '')" >> ${CMAKE_BINARY_DIR}/setup.fish VERBATIM
  COMMAND echo "if  test -z \"\$pyversion\" " >> ${CMAKE_BINARY_DIR}/setup.fish VERBATIM
  COMMAND echo "    set -l PYTHON_EXE python3" >> ${CMAKE_BINARY_DIR}/setup.fish VERBATIM
  COMMAND echo "    set -l pyversion (python3 -V 2>&1 | grep -Po '(?<=Python )(.+)' || echo '')" >> ${CMAKE_BINARY_DIR}/setup.fish VERBATIM
  COMMAND echo "    if test -z \"\$pyversion\" ]]" >> ${CMAKE_BINARY_DIR}/setup.fish VERBATIM
  COMMAND echo "    echo \"No Python is detected! Cannot source\" " >> ${CMAKE_BINARY_DIR}/setup.fish VERBATIM
  COMMAND echo "    exit" >> ${CMAKE_BINARY_DIR}/setup.fish VERBATIM
  COMMAND echo "    end" >> ${CMAKE_BINARY_DIR}/setup.fish VERBATIM
  COMMAND echo "end" >> ${CMAKE_BINARY_DIR}/setup.fish VERBATIM
  COMMAND echo "set SCRIPT (\$PYTHON_EXE -c \"import os,sys; print(os.path.realpath(os.path.expanduser(sys.argv[1])))\" \$SCRIPT_FILE)" >> ${CMAKE_BINARY_DIR}/setup.fish VERBATIM
  COMMAND echo "set -gx IRIS (dirname -- \$SCRIPT)" >> ${CMAKE_BINARY_DIR}/setup.fish
  COMMAND echo "set -gx LD_LIBRARY_PATH . \$IRIS/lib \$IRIS/lib64 \$LD_LIBRARY_PATH" >> ${CMAKE_BINARY_DIR}/setup.fish
  COMMAND echo "set -gx CPATH \$IRIS/include \$CPATH" >> ${CMAKE_BINARY_DIR}/setup.fish
  COMMAND echo "set -gx PYTHONPATH \$IRIS \$PYTHONPATH" >> ${CMAKE_BINARY_DIR}/setup.fish
  COMMAND echo "set -gx SYSTEM (hostname|cut -d . -f 1|sed 's/[0-9]*//g')" >>  ${CMAKE_BINARY_DIR}/setup.fish
  COMMAND echo "set -gx MACHINE \${SYSTEM^}" >>  ${CMAKE_BINARY_DIR}/setup.fish
  COMMAND echo "echo \"*********************** IRIS Environent *********************\"" >> ${CMAKE_BINARY_DIR}/setup.fish
  COMMAND echo "echo \"IRIS: \$IRIS\"" >> ${CMAKE_BINARY_DIR}/setup.fish
  COMMAND echo "echo \"*************************************************************\"" >> ${CMAKE_BINARY_DIR}/setup.fish
  )
add_custom_target(setup_source ALL DEPENDS ${CMAKE_BINARY_DIR}/setup.source)
add_custom_target(setup_zsh ALL DEPENDS ${CMAKE_BINARY_DIR}/setup.zsh)
add_custom_target(setpu_fish ALL DEPENDS ${CMAKE_BINARY_DIR}/setup.fish)
install(FILES ${CMAKE_BINARY_DIR}/setup.source DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES ${CMAKE_BINARY_DIR}/setup.zsh DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES ${CMAKE_BINARY_DIR}/setup.fish DESTINATION ${CMAKE_INSTALL_PREFIX})
