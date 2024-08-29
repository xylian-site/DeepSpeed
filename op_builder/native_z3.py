# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import TorchCPUOpBuilder


class NativeZ3Builder(TorchCPUOpBuilder):
    BUILD_VAR = "DS_BUILD_NATIVE_Z3"
    NAME = "native_z3"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.compile.{self.NAME}_op'

    def sources(self):
        return ['csrc/compile/native_z3.cpp']

    def libraries_args(self):
        args = super().libraries_args()
        return args

    def include_paths(self):
        return ['csrc/includes']
