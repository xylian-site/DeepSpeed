import torch
import networkx as nx
from .nx import fx_to_nx
from .fx import get_output_node
from .graph_param import DSGraphParamManager


def get_nx_output_node(G: nx.DiGraph):
    for node in G.nodes:
        if node.op == "output":
            return node
    raise ValueError("No output node found")


def find_release_nodes(G: nx.DiGraph):
    release_nodes = []
    for node in G.nodes:
        if node.name.startswith("release_param"):
            release_nodes.append(node)
    return release_nodes


def find_all_dependency_nodes(G: nx.DiGraph, node, dependency_graph=None) -> nx.DiGraph:
    if dependency_graph is None:
        dependency_graph = nx.DiGraph()
    
    # add node to dependency graph if not already present
    if node in dependency_graph.nodes:
        return
    
    dependency_graph.add_node(node)
    for pred in G.predecessors(node):
        find_all_dependency_nodes(G, pred, dependency_graph)
        dependency_graph.add_edge(pred, node)
        
    return dependency_graph


def sum_allgather_sizes(G: nx.DiGraph, param_manager: DSGraphParamManager) -> int:
    allgather_sizes = 0
    for node in G.nodes:
        if param_manager.is_allgather_node(node):
            param_name = param_manager.allgather_param_name(node)
            graph_param = param_manager.get_graph_param(param_name)
            allgather_sizes += graph_param.numel
    return allgather_sizes


def schedule_by_distance(G: nx.DiGraph) -> nx.DiGraph:
    unscheduled = list(G.nodes)
    scheduled = []

    # schedule placeholder first
    for node in unscheduled:
        if node.op == "placeholder":
            scheduled.append(node)
            unscheduled.remove(node)

    print("unscheduled", unscheduled)
    print("scheduled", scheduled)

    output_node = get_output_node(G)
    distances = nx.single_source_shortest_path_length(G.reverse(), output_node)

    i = 0
    while len(unscheduled) > 0:

        schedulable = []
        for node in unscheduled:
            if all([pred in scheduled for pred in G.predecessors(node)]):
                schedulable.append(node)

        print(f"{i}: schedulable {schedulable}")

        next_node = None
        for node in schedulable:
            if node.name.startswith("release_param"):
                print(f"Chose release node {node}")
                next_node = node
            # elif node.name.startswith("allgather_param"):
            #     print(f"Chose gather node {node}")
            #     next_node = node
        if next_node is None:
            # sort schedulable nodes by distance to output
            schedulable = sorted(schedulable, key=lambda node: distances[node], reverse=True)
            next_node = schedulable[0]

        scheduled.append(next_node)
        unscheduled.remove(next_node)
        schedulable.remove(next_node)

        print(f"{i}: selected {next_node}")

        # for node in schedulable:
        #     scheduled.append(node)
        #     unscheduled.remove(node)

        print(f"{i}: unscheduled {unscheduled}")
        print(f"{i}: scheduled {scheduled}")

        i += 1


def schedule(fx_graph: torch.fx.Graph, param_manager: DSGraphParamManager) -> torch.fx.Graph:
    output_node = get_output_node(fx_graph)
    nx_graph = fx_to_nx(fx_graph)

    distances = nx.single_source_shortest_path_length(nx_graph.reverse(), output_node)
    for n, d in distances.items():
        print(f"n={n} {n.op} distance={d}")

    schedule_by_distance(nx_graph)

    for n in find_release_nodes(nx_graph):
        param_name = param_manager.release_param_name(n)
        graph_param = param_manager.get_graph_param(param_name)
        print(f"release node {n}: param_name={param_name} numel={graph_param.numel}")
        dependencies = find_all_dependency_nodes(nx_graph, n)
        print(f"  dependency {dependencies} sum_allgather={sum_allgather_sizes(dependencies, param_manager)}")

    return fx_graph
