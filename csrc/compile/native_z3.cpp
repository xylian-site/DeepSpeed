// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "native_z3.h"

// C++ interface

void test_call(at::Tensor param)
{
    std::cout << "test_call " << param << std::endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("test_call",
          &test_call,
          "Test function");
}
