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

[unreleased]: https://github.com/ORNL/iris/compare/v3.0.0...HEAD
[3.0.0]: https://github.com/ORNL/iris/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/ORNL/iris/compare/v1.0.1...v2.0.0
[1.0.1]: https://github.com/ORNL/iris/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/ORNL/iris/releases/tag/v1.0.0
