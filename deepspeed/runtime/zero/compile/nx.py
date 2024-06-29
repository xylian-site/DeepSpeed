import torch.fx as fx
import networkx as nx

def fx_to_nx(fx_graph):
    """Converts a torch.fx.Graph to a NetworkX graph."""
    nx_graph = nx.DiGraph()

    def traverse(node: fx.Node):
        if node in nx_graph:
            return
        nx_graph.add_node(node)

        for next_node in node.users.keys():
            nx_graph.add_edge(node, next_node)
            traverse(next_node)

    for node in fx_graph.nodes:
        traverse(node)

    from nxpd import draw
    draw(nx_graph, filename="nx_graph.png", format="png")

    return nx_graph


def find_reachable_terminal_nodes(G, marked_nodes):
    qualifying_nodes = []
    
    for node in marked_nodes:
        paths = nx.single_source_shortest_path(G, node)
        terminal_nodes = [n for n in G.nodes if G.out_degree(n) == 0]
        
        valid_terminal_nodes = []
        for terminal_node in terminal_nodes:
            path = paths.get(terminal_node, [])
            if path and not any(mark in path[1:-1] for mark in marked_nodes):
                valid_terminal_nodes.append(terminal_node)
        
        if valid_terminal_nodes:
            qualifying_nodes.append(node)
    
    return qualifying_nodes