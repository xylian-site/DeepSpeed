// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "native_z3.h"

// C++ interface

at::Tensor test_call(at::Tensor param)
{
    std::cout << "test_call " << param << std::endl;
    return param;
}

TORCH_LIBRARY(native_z3, m)
{
    // Note that "float" in the schema corresponds to the C++ double type
    // and the Python float type.
    m.def("test_call(Tensor a) -> Tensor");
}

TORCH_LIBRARY_IMPL(native_z3, CPU, m) { m.impl("test_call", &test_call); }

TORCH_LIBRARY_IMPL(native_z3, CUDA, m) { m.impl("test_call", &test_call); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("test_call", &test_call, "Test function"); }
