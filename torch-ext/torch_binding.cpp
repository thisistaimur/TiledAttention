#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("tiledattention(Tensor! out, Tensor input) -> ()");
#if defined(CPU_KERNEL)
  ops.impl("tiledattention", torch::kCPU, &tiledattention);
#elif defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  ops.impl("tiledattention", torch::kCUDA, &tiledattention);
#elif defined(METAL_KERNEL)
  ops.impl("tiledattention", torch::kMPS, tiledattention);
#elif defined(XPU_KERNEL)
  ops.impl("tiledattention", torch::kXPU, &tiledattention);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)