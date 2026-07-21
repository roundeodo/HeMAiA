import pathlib
import sys

import networkx as nx


MINI_COMPILER = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MINI_COMPILER))

from bingo_dfg import BingoDFG
from bingo_node import BingoNode


def make_node(graph, cluster, core):
    node = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=cluster,
        assigned_core_id=core,
        kernel_name=None,
    )
    graph.bingo_add_node(node)
    return node


def test_dummy_sets_precede_next_task_on_producer_resource():
    graph = BingoDFG(
        num_chiplets=1,
        num_clusters_per_chiplet=3,
        num_cores_per_cluster=3,
        is_host_as_acc=False,
        chiplet_ids=[0],
    )

    producer = make_node(graph, cluster=0, core=2)
    shared_relay = make_node(graph, cluster=0, core=1)
    later_same_resource = make_node(graph, cluster=0, core=2)
    remote_consumer_a = make_node(graph, cluster=1, core=1)
    remote_consumer_b = make_node(graph, cluster=2, core=1)

    graph.add_edges_from(
        [
            (producer, shared_relay),
            (shared_relay, later_same_resource),
            (producer, remote_consumer_a),
            (producer, remote_consumer_b),
        ]
    )

    graph.bingo_transform_add_resource_sequencing_edges()
    graph.bingo_transform_dfg_add_dummy_set_nodes()
    graph.bingo_transform_chain_dummy_sets_before_resource_successors()
    graph.bingo_transform_dfg_add_dummy_check_nodes()
    graph.bingo_assign_normal_node_dep_set_info()
    graph.bingo_assign_normal_node_dep_check_info()
    graph.bingo_transform_dfg_allocate_dep_tags(tag_width=2)

    dummy_sets = [
        node
        for node in graph.nodes
        if node.node_type == "dummy"
        and node.dep_set_enable
        and not node.dep_check_enable
        and node.assigned_cluster_id == producer.assigned_cluster_id
        and node.assigned_core_id == producer.assigned_core_id
    ]
    assert len(dummy_sets) == 2
    assert all(
        nx.has_path(graph, dummy, later_same_resource)
        for dummy in dummy_sets
    )

    emitted = graph._resource_balanced_topological_sort(0)
    position = {node: index for index, node in enumerate(emitted)}
    assert all(
        position[dummy] < position[later_same_resource]
        for dummy in dummy_sets
    )
    assert later_same_resource.dep_check_enable
    assert later_same_resource.dep_check_list == [shared_relay.assigned_core_id]


def test_descriptor_sequence_edges_do_not_encode_scoreboard_dependencies():
    graph = BingoDFG(
        num_chiplets=1,
        num_clusters_per_chiplet=1,
        num_cores_per_cluster=2,
        is_host_as_acc=False,
        chiplet_ids=[0],
    )

    first = make_node(graph, cluster=0, core=1)
    second = make_node(graph, cluster=0, core=1)

    graph.add_edge(first, second, descriptor_sequence=True)

    graph.bingo_transform_dfg_add_dummy_set_nodes()
    graph.bingo_transform_dfg_add_dummy_check_nodes()
    graph.bingo_assign_normal_node_dep_set_info()
    graph.bingo_assign_normal_node_dep_check_info()
    graph.bingo_transform_dfg_allocate_dep_tags(tag_width=1)

    assert not first.dep_set_enable
    assert not second.dep_check_enable
