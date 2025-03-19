# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

backward_inputs = []

enabled_patched_func = False
original_grad_fn = None
base_meta = type(torch.autograd.Function)

from deepspeed.utils.torch import required_torch_version
if required_torch_version(min_version=2.6):

    class FunctionMeta(base_meta):

        def __new__(cls, name, bases, dct):
            if name == "CompiledFunction":
                original_backward_prologue = dct.get("_backward_prologue")

                def wrapped_backward_prologue(ctx, *grad_outputs):
                    assert original_backward_prologue is not None

                    all_args = original_backward_prologue(ctx, *grad_outputs)
                    if enabled_patched_func:
                        backward_inputs.append(all_args)
                        wrapped_backward_prologue.owner_class.compiled_bw = None

                    return all_args

                wrapped_backward_prologue.owner_class = None
                dct["_backward_prologue"] = staticmethod(wrapped_backward_prologue)
                new_class = super().__new__(cls, name, bases, dct)
                wrapped_backward_prologue.owner_class = new_class

                return new_class

            return super().__new__(cls, name, bases, dct)


def patch_compiled_func():

    global enabled_patched_func
    enabled_patched_func = True

    class PatchedFunction(torch.autograd.Function, metaclass=FunctionMeta):
        pass

    global original_grad_fn
    original_grad_fn = torch.autograd.Function
    torch.autograd.Function = PatchedFunction

    return backward_inputs


def unpatch_compiled_func():
    global enabled_patched_func
    enabled_patched_func = False

    global original_grad_fn
    torch.autograd.Function = original_grad_fn


def get_backward_inputs():
    return backward_inputs
