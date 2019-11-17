
def preprocess(graph):
    all_assert_nodes = graph.find_nodes_by_name("Equal")
    graph.remove(all_assert_nodes, remove_exclusive_dependencies=False)

    all_assert_nodes = graph.find_nodes_by_op("Select")
    gather = []
    for node in all_assert_nodes:
        if "PadOrClipBoxList" in node.name:
            gather.append(node)
    graph.remove(gather, remove_exclusive_dependencies=False)

    all_assert_nodes = graph.find_nodes_by_op("Greater")
    gather = []
    for node in all_assert_nodes:
        if "PadOrClipBoxList" in node.name:
            gather.append(node)
    graph.remove(gather, remove_exclusive_dependencies=False)

    return graph
