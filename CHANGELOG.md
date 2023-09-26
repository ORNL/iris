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

### Added
- Changelog added to the repo to track major changes.

### Changed

### Removed

### Fixed

## [2.0.0] - 2023-06-02

Major release.

## [1.0.1] - 2023-06-02

Moved documentation from external repo to be part of the code repo.

## [1.0.0] - 2023-06-02

Initial Release of IRIS, an intelligent runtime system for extremely heterogeneous architectures. IRIS discovers available functionality, manage multiple diverse programming
systems (e.g., OpenMP, CUDA, HIP, Level Zero, OpenCL, Hexagon) simultaneously in the same application, represents data dependencies, orchestrates data movement proactively, and allows configurable work schedulers for diverse heterogeneous devices.