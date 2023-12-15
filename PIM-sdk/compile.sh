#!/bin/bash

../../pim-iree-build/tools/iree-compile --iree-input-type=mhlo --mlir-disable-threading --mlir-print-ir-after-all --iree-hal-target-backends=pim --compile-to=end ./compiler-artifact/gpt_125M.mlir -o ./compiler-artifact/gpt_125M_end_pim.vmfb

