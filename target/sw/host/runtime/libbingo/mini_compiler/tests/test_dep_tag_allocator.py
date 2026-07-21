import pathlib
import sys
from types import MethodType


MINI_COMPILER = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MINI_COMPILER))

from bingo_dfg import BingoDFG
from bingo_node import BingoNode


def make_node(graph, cluster=0, core=1):
    node = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=cluster,
        assigned_core_id=core,
        kernel_name=None,
    )
    graph.bingo_add_node(node)
    return node


def test_tag_reuse_follows_emitted_descriptor_order():
    graph = BingoDFG(
        num_chiplets=1,
        num_clusters_per_chiplet=1,
        num_cores_per_cluster=2,
        is_host_as_acc=False,
        chiplet_ids=[0],
    )

    start_a = make_node(graph, core=0)
    start_b = make_node(graph, core=0)
    producer_a = make_node(graph)
    consumer_a = make_node(graph)
    middle_b = make_node(graph, core=0)
    producer_b = make_node(graph)
    consumer_b = make_node(graph)

    graph.add_edges_from(
        [
            (start_a, producer_a),
            (producer_a, consumer_a),
            (start_b, middle_b),
            (middle_b, producer_b),
            (producer_b, consumer_b),
            (consumer_a, consumer_b),
        ]
    )

    for producer in (producer_a, producer_b):
        producer.dep_set_enable = True
        producer.dep_set_list = [1]
        producer.dep_set_chiplet_id = 0
        producer.dep_set_cluster_id = 0
    for consumer in (consumer_a, consumer_b):
        consumer.dep_check_enable = True
        consumer.dep_check_list = [1]

    # Both producers enter the resource before either consumer can clear its
    # token. Reusing one presence-bit tag here would lose one dependency.
    emitted = [
        start_a,
        start_b,
        middle_b,
        producer_a,
        producer_b,
        consumer_a,
        consumer_b,
    ]
    graph._resource_balanced_topological_sort = MethodType(
        lambda self, chiplet_id: emitted, graph
    )

    graph.bingo_transform_dfg_allocate_dep_tags(tag_width=1)

    assert producer_a.dep_set_tag == consumer_a.dep_check_tag
    assert producer_b.dep_set_tag == consumer_b.dep_check_tag
    assert producer_a.dep_set_tag != producer_b.dep_set_tag


if __name__ == "__main__":
    test_tag_reuse_follows_emitted_descriptor_order()
