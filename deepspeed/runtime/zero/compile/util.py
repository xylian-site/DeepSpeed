from torch.fx import Node

def is_comm_op(node: Node) -> bool:
    return hasattr(node.meta, "comm") and node.meta["comm"]
