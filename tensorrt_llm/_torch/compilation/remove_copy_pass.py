# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from operator import getitem

import torch
from torch._higher_order_ops.auto_functionalize import (auto_functionalized,
                                                        auto_functionalized_v2)
from torch.fx import Graph, Node

from .utils import (EAGLE_HIDDEN_STATES_CAPTURE_IDX_META, inplace_info,
                    is_call_function)

aten = torch.ops.aten


def remove_copy_for_mutates_args(graph: Graph):
    '''
    This pass is to remove the copy nodes for the mutates args introduced by the auto_functionalize.
    '''

    nodes_to_remove: list[Node] = []

    def remove_functionalize_inner(node: Node, mutates_args: dict, is_v2=False):
        getitem_nodes = [
            user for user in node.users if is_call_function(user, getitem)
        ]

        kwargs = {k: v for k, v in node.kwargs.items() if not k.startswith("_")}
        if is_v2:
            all_bases = node.kwargs["_all_bases"]
            for arg in inplace_func._schema.arguments:
                if arg.alias_info is None or not arg.alias_info.is_write:
                    continue
                base_index_key = f"_{arg.name}_base_index"
                base_index = node.kwargs[base_index_key]
                kwargs[arg.name] = (None if base_index is None else
                                    all_bases[base_index])

        for getitem_node in getitem_nodes:
            idx = getitem_node.args[1]
            mutated_arg = mutates_args[idx]
            replacement = kwargs[mutated_arg]
            assert replacement is not None, (
                f"getitem user for optional output '{mutated_arg}' "
                "has no base tensor -- graph is malformed")
            getitem_node.replace_all_uses_with(replacement)
            nodes_to_remove.append(getitem_node)

        with graph.inserting_before(node):
            graph.call_function(inplace_func, kwargs=kwargs)

        nodes_to_remove.append(node)

    for node in graph.nodes:
        if not is_call_function(node, [
                auto_functionalized,
                auto_functionalized_v2,
        ]):
            continue

        inplace_func = node.args[0]

        inplace_map = inplace_info()
        if inplace_func not in inplace_map:
            # We do not know the inplace op
            continue

        remove_functionalize_inner(
            node,
            inplace_map[inplace_func],
            is_v2=node.target == auto_functionalized_v2,
        )

    for node in nodes_to_remove:
        graph.erase_node(node)


def extract_eagle_hidden_state_publications(graph: Graph) -> None:
    """Fold Eagle publication markers into their physical buffer writes.

    The marker is declared mutable so AOTAutograd keeps it through dead-code
    elimination. Leaving that mutation in the graph until piecewise splitting
    changes submodule boundaries and, consequently, multi-stream scheduling.
    Convert it to producer metadata here, after DCE but before piecewise split,
    so the executable graph seen by the scheduler is otherwise unchanged.
    """
    latest_inplace_stat: dict[Node, Node] = {}
    nodes_to_remove: list[Node] = []
    inplace_map = inplace_info()
    publication_op = torch.ops.trtllm.publish_eagle_hidden_states.default
    slice_copy_op = torch.ops.trtllm.inplace_slice_copy.default

    for node in graph.nodes:
        if is_call_function(node, publication_op):
            hidden_states_buffer = node.kwargs["hidden_states_buffer"]
            capture_idx = node.kwargs["capture_idx"]
            producer = latest_inplace_stat.get(hidden_states_buffer)
            assert producer is not None, (
                "Eagle hidden-state publication must follow an in-place "
                "buffer write")
            assert is_call_function(
                producer,
                slice_copy_op), ("Eagle hidden-state publication must follow "
                                 "inplace_slice_copy")
            assert EAGLE_HIDDEN_STATES_CAPTURE_IDX_META not in producer.meta
            producer.meta[EAGLE_HIDDEN_STATES_CAPTURE_IDX_META] = capture_idx
            nodes_to_remove.append(node)
            continue

        if node.op != "call_function" or node.target not in inplace_map:
            continue
        for inplace_arg in inplace_map[node.target].values():
            assert inplace_arg in node.kwargs
            mutated_arg = node.kwargs[inplace_arg]
            if isinstance(mutated_arg, Node):
                latest_inplace_stat[mutated_arg] = node

    for node in nodes_to_remove:
        assert not node.users
        graph.erase_node(node)
