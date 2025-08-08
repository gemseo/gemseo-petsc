<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

<!--
Changelog titles are:
- Added: for new features.
- Changed: for changes in existing functionality.
- Deprecated: for soon-to-be removed features.
- Removed: for now removed features.
- Fixed: for any bug fixes.
- Security: in case of vulnerabilities.
-->

# Changelog

All notable changes of this project will be documented here.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0)
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Version 4.1.1 (August 2025)

## Fixed

- Initialisation of PETSc.

## Version 4.1.0 (March 2025)

## Added

- Implementation of the interface between PETSc.TS and Gemseo for the solution of Ordinary Differential Equations and computation of the adjoint.

## Version 4.0.0 (November 2024)

### Added

- Support GEMSEO v6.
- Support for Python 3.12.

### Changed

- Renamed `ksp_library.py` to `petsc_ksp.py`
- Switched from JSON schema to Pydantic model for options validation.
- Each solver now has a dedicated GEMSEO algo, named `PETSC_...`. This replaces the
  `solver_type` option.

## Version 3.0.1 (December 2023)

### Added

- Support for Python 3.11.

### Removed

- Support for Python 3.8.

## Version 3.0.0 (June 2023)

Update to GEMSEO 5.0.0.

### Changed

Renamed `ksp_lib.py` to `ksp_library.py`.

## Version 2.0.0 (November 2021)

Update to GEMSEO 4.0.0.

## Version 1.0.0 (November 2021)

First release.
