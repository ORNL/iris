# Changelog

All notable changes to IRIS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to a custom Development Versioning specified by Aaron Young.

A summary of Development Versioning Specification is shown below.

> Given a version number BRANCH.TAG.BUILD, increment the:
> 1. BRANCH version when you make breaking/major changes that you want to track in a separate branch.
> 2. TAG version when you make a new tag to mark a specific spot.
> 3. BUILD version when you create a new build with artifacts or bug fixes for that you want to point to.
>
> Then for your repo you have branch versions for each version. For example branches v0 and v1. Then when you create tags, say on branch v0, you would create tags v0.0.0, v0.1.0, and v0.2.0.
> CI or a manual process could add v0.0.x branches as new changes are added to a local branch. BUILD is also used when patches are applied to a tagged branch, after the patch is applied, add a new tag with BUILD + 1.
>
> `main` always points to the current major branch plus 1. `dev` is an integration branch before merging into `main`. When `dev` is merged into `main`, the TAG is updated.

## [Unreleased]

## [4.0.0] - 2026-11-03

### Added

#### New Capabilties

- Q-IRIS: integration with QIREE/QSIM/Qiskit for quantum kernel execution, including process-based QSIM/QIREE execution (for thread-safety), qcut QIR circuit test suite, and helpers to interpret QIREE/qiskit results
- Expanded Julia support: subarray view handling, 2D/3D native kernels, `parallel_reduce` (sum/prod/maximum/minimum) with output/temporary buffer reuse, DMEM caching, custom task/kernel policies, and copy/offset/dev_size APIs
- Flexible DMEM region support: 3-way region protocol, partial region access, control over number of valid regions, and dynamic GWS/argument updates for kernel and copy commands
- Conditional task execution and Graph resubmission (loop until a condition is met or a maximum iteration count is reached), plus a Graph end-task fetch API
- Device-specific malloc/free APIs and APIs to probe whether a task or memory object is still alive
- Support for naming tasks and memory objects (`iris_task_set_name`/`iris_task_get_name`), and an API to fetch the last kernel command of a task
- Dynamic profiling controls: enable/disable profiling per task/graph (`iris_set_enable_profile`), DOT profiler controls, and draft profiling support for mixed-precision kernels
- Apple/macOS support: initial compilation support, default compiler selection, and OpenMP-based threading
- Python: `@ptask` decorator for simplified task creation, and a unified spec-based interface standardizing kernel calls across Python, host (CUDA/HIP), and Julia
- Runtime loading of CUDA modules, and `IRIS_RUNTIME_KERNELS_PATH` environment variable

### Changed

- D2D CUDA/HIP memory copies now use `cuMemcpyPeer`/equivalent instead of `cuMemcpyD2D`/`hipMemcpyD2D` to avoid cross-context issues
- Malloc-async is now disabled by default (its interaction with D2D across contexts was unreliable); opt back in with `IRIS_MALLOC_ASYNC=1`
- CUDA/HIP memory copies support 3D/2D transfers with strides, and 2D copies that are contiguous are converted to 1D
- Default CUDA/HIP threads-per-block is now `min(256, size)`
- C++14 is now the required minimum standard
- Proactive/lazy dispatch logic reworked at the scheduler and worker; proactive dispatch remains disabled by default pending further fixes
- Python default GWS in the task builder updated to `[1]`
- CI: adjusted module loading (no longer loads `cmake`/`gnu` modules), refined error/warning log matching, and tuned job timeouts

### Removed

- Redundant `Release` API from the IRIS Python interface
- Duplicate Julia math APIs (`inner`, `dot`)

### Fixed

- Several DMEM/region bugs: uninitialized source FILL state in DMEM-to-DMEM copies, reset/refresh/flush issues, and async issues when a kernel writes to a full DMEM with regions
- Multiple proactive-dispatch bugs, including multi-device dispatch and event-dependency corner cases
- Queue-vs-scheduler cleanup ordering bug that could delete the queue before the scheduler
- Several memory leaks and logic bugs identified via code review
- Various build/compilation fixes across CMake, CUDA, HIP, Julia, Python, Xilinx, Zenith, and macOS
- OpenCL async mode now queries whether the host command queue supports out-of-order execution before requesting it, falling back to an in-order queue when unsupported; fixes crashes/hangs on devices (e.g. AMD gfx1030) whose host queues reject `CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE`

## [3.0.0] - 2025-05-30

### Added

#### New Capabilties

- Asynchronous task execution and utilizes underlying device specific asynchronous multi-stream capabilities
- Handle stream synchronization within device and avoid host involvement
- Overlapped computation with data transfer using device provided asynchronous multi-stream capabilities
- Also give all streams to host wrapper of (CUDA/HIP) kernel if exists, which can explore all streams
- Automatic dependency identification based on data flow analysis
- Added Tiling data structures
- Added initial Julia support with IRIS Julia APIs to call kernels written in native language
- New Retain-Release logic based tasks garbage collection
- Advanced Dagger features and testing
- Task Graph policy
- Changelog added to the repo to track major changes.
- Xilinx FPGA kernel acceleration support
- FFI-based kernel calls (No need of boiler plate. To be enabled using ENABLE_FFI)
- Added multi-purpose HostInterface for traditional boiler-plate based host kernel calls and FFI-based calls
- Generalized Loader for loading shared libraries of kernels
- Support for UniSYCL (In external repository)

#### Updates

- Also enabled manual control of DMEM (H2D, D2H) through IRIS APIs along with automatic data orchestration
- Added DMEM to DMEM copy command support

### Changed

- Improved Python interface and Python PIP package installation
- Improved C++ IRIS interface with DMEM type specific templates
- iris_task and iris_mem are no longer pointers. They are structs now and the memory will be in application's memory space.

### Removed

- None

### Fixed

- Several bug fixes and fixes for deadlock issues
- DMEM multiple PIN memory registrations
- Fixed fortan host
- Fixed OpenCL device loading issues
- iris_task and iris_mem are no longer pointers. They are structs now and the memory will be in application's memory space.

## [2.0.0] - 2023-06-02

Major release.

## [1.0.1] - 2023-06-02

Moved documentation from external repo to be part of the code repo.

## [1.0.0] - 2023-06-02

Initial Release of IRIS, an intelligent runtime system for extremely heterogeneous architectures. IRIS discovers available functionality, manage multiple diverse programming
systems (e.g., OpenMP, CUDA, HIP, Level Zero, OpenCL, Hexagon) simultaneously in the same application, represents data dependencies, orchestrates data movement proactively, and allows configurable work schedulers for diverse heterogeneous devices.

[unreleased]: https://github.com/ORNL/iris/compare/v4.0.0...HEAD
[4.0.0]: https://github.com/ORNL/iris/compare/v3.0.0...v4.0.0
[3.0.0]: https://github.com/ORNL/iris/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/ORNL/iris/compare/v1.0.1...v2.0.0
[1.0.1]: https://github.com/ORNL/iris/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/ORNL/iris/releases/tag/v1.0.0
