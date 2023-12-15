---

# PIM-IREE

This project is a compiler and runtime for executing GLM models on PIM devices. (It is a fork of [IREE](https://github.com/aiha-lab/iree.git)). A model described in MLIR's StableHLO dialect is taken as input and compiled into code that can operate on a PIM device, and it can be executed by connecting with the SDK through the runtime.

## Installation Instructions

### Download pim-iree

```bash
git clone https://github.com/aiha-lab/pim-iree.git
cd pim-iree
git submodule update --init
```

### Install Jsoncpp

Reference: [Jsoncpp GitHub](https://github.com/open-source-parsers/jsoncpp.git)

```bash
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg integrate install
./vcpkg install jsoncpp
```

### CMake Build

```bash
cmake -G Ninja -B ../pim-iree-build/ . -DCMAKE_TOOLCHAIN_FILE=[vcpkg root]/scripts/buildsystems/vcpkg.cmake -DCMAKE_PREFIX_PATH="[vcpkg root]/installed/x64-linux"
cmake --build ../pim-iree-build/
```

## Usage

### Compile

```bash
cd pim-iree/PIM-sdk/
./compile.sh
```

Input file is a GPT2-125M model described in the StableHLO dialect. The final output of the compiler is a vmfb file, and if you want to check the intermediate result(MLIR), you can modify and use the --compile-to option in compile.sh.

### Runtime Execution

```bash
./exec-pim.sh
```
