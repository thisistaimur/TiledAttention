#pragma once

#include <torch/torch.h>

void tiledattention(torch::Tensor &out, torch::Tensor const &input);