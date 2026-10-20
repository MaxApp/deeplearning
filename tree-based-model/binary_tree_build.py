from collections import deque
from typing import Any, Optional, Sequence


class TreeNode:
    def __init__(self, value: Any) -> None:
        self.value = value
        self.left: Optional["TreeNode"] = None
        self.right: Optional["TreeNode"] = None


def build_binary_tree(level_order: Sequence[Optional[Any]]) -> Optional[TreeNode]:
    if not level_order:
        return None

    if level_order[0] is None:
        if any(value is not None for value in level_order[1:]):
            raise ValueError("Empty root error")
        return None

    root = TreeNode(level_order[0])
    level_queue = deque([root])
    index = 1

    while level_queue and index < len(level_order):
        parent = level_queue.popleft()

        left_value = level_order[index]
        index += 1
        if left_value is not None:
            parent.left = TreeNode(left_value)
            level_queue.append(parent.left)

        if index < len(level_order):
            right_value = level_order[index]
            index += 1
            if right_value is not None:
                parent.right = TreeNode(right_value)
                level_queue.append(parent.right)

    if any(value is not None for value in level_order[index:]):
        raise ValueError("The level-order sequence contains nodes without parents.")

    return root


if __name__ == "__main__":
    values = [1, 2, 3, None, 5]
    root = build_binary_tree(values)
