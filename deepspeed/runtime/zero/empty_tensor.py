import torch
from torch.utils._pytree import tree_map
import contextlib


@contextlib.contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


class NonRewrappingTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls, t: torch.Tensor
    ):
        r = super()._make_wrapper_subclass(
            cls, t.shape, dtype=t.dtype, requires_grad=t.requires_grad, device=t.device)
        return r

    def __init__(self, t) -> None:
        self.tensor: torch.Tensor = t

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):

        def unwrap(e) -> torch.Tensor:
            if isinstance(e, NonRewrappingTensor):
                t = e.tensor
                return t
            else:
                return e

        r = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
        # Return an unwrapped tensor no longer of original subclass type.
        return r


tensor_map = {}

class EmptyTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem):
        obj = torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            dtype=elem.dtype,
            layout=elem.layout,
            requires_grad=elem.requires_grad,
            device=elem.device,
        )

        return obj

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def inflate(t):
            if isinstance(t, cls):
                with no_dispatch():
                    return torch.ones_like(t, device=t.device)
            else:
                return t

        args = tree_map(inflate, args)
        kwargs = tree_map(inflate, kwargs)

        # https://github.com/pytorch/pytorch/issues/77265
        if func is torch.ops.aten.detach.default:
            # Special handling for detach
            result = args[0].detach()
            return EmptyTensor(result)

        ret = super().__torch_dispatch__(
                func, types, args, kwargs
        )

        return ret
