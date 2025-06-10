# Leela Chess Zero (LCZero) Codebase Overview

## Building and running

### Building

* Build system is Meson.
* Debug builds are located in `builddir/` (as VS Code usually creates them). That's where we work on the code most of the time, unless we optimize performance.
* Release builds are located in `build/release/`
* Default set of build options is reasonable for most cases, but to examine all options, check `meson_options.txt` and `meson.build` files.
* To build, cd into the build directory and run `ninja lc0`.
* To test, run `ninja test` (it will build more dependencies, so longer than `ninja lc0`).

### Running

* To run, `./lc0 [search-algorithm] [flags]`. It will run the UCI engine.
* Instead of search algorithm, other tools like `benchmark`, `onnx2leela`, `leela2onnx`, `describenet`, `benchmark`, `backendbench`, `selfplay` can be used.
* Flags can be queried with `./lc0 --help`. It's also possible to set them through UCI options.
* In order to run the engine, you need to have a network file. Users often have somewhere on their system, but if not, download <https://storage.lczero.org/files/networks-contrib/t3-512x15x16h-distill-swa-2767500.pb.gz> into the engine directory.
* The engine will automatically discover the network file and use it.

## Code structure

* `src/search/` - Contains search algorithms:
  * `classic` in `src/search/classic/` - The main algorithm. Quite messy.
  * `dag-preview` in `src/search/dag_classic/` - fork of the classic algorithm with a DAG support.
  * `lc3` in `src/search/lc3/` - Experimental search algorithm in development.
  * Algorithms implement interfaces in `src/search/search.h`.
  * Algorithms are registered using `REGISTER_SEARCH` macro.
* `src/neural/` - Contains neural network backends which evaluate positions:
  * `src/neural/network.h` - old interface, registered using `REGISTER_NETWORK` macro.
  * `src/neural/backend.h` - new interface, registered using `REGISTER_BACKEND` macro.
  * Search algorithms use new interface, but most, if not all, backends still implement the old interface. `src/neural/wrapper.{h,cc}` provides a wrapper to use old backends with the new interface.
  * grep for `REGISTER_NETWORK` and `REGISTER_BACKEND` to find all backends.
  * Some backends (`check`, `demux`, `multiplexing`, `roundrobin`) are not real backends, but rather wrappers to use multiple backends at once.
* `src/chess/` - Contains chess logic (position, movegen, etc.):
* `src/utils/` - Contains utility code (logging, exceptions, etc.):
* `src/tools/`, `src/selfplay` - Contains tools (benchmark, network conversion, etc.):

## Coding standards

* Most of C++20 is supported/encouraged.
  * Particularly, constructors that take structs with designated initializers are preferred.
* Google C++ Style Guide is used, with the following notes:
  * Use `#pragma once` instead of header guards.
  * Use `lczero::Exception` for exceptions, no other exceptions allowed. No much exception safety is needed.
  * Non-const reference function parameters are not encouraged (over pointers) nor discouraged.
