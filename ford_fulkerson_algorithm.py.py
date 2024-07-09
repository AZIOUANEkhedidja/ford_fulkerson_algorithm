import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

# Function to read adjacency matrix from file
def read_adjacency_matrix(file_path):
    """
    Reads an adjacency matrix from a text file.

    Args:
    - file_path (str): Path to the file containing the adjacency matrix.

    Returns:
    - list of list of int: The adjacency matrix represented as a 2D list.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()
    adjacency_matrix = []
    adjacency_matrix.append(lines[0].strip().split())
    for line in lines[1:]:
        row_data = line.strip().split()
        row = [int(value) for value in row_data[1:]]
        adjacency_matrix.append(row)
    return adjacency_matrix[1:]

# Function to display residual graphs using matplotlib and networkx
def show_graphs(residual_graphs):
    """
    Displays the residual graphs using matplotlib and networkx.

    Args:
    - residual_graphs (list of list of int): List of residual graphs represented as adjacency matrices.
    """
    plt.figure(figsize=(12, 8))
    for idx, graph in enumerate(residual_graphs, start=1):
        G = residual_to_networkx(graph)
        plt.subplot(2, (len(residual_graphs) + 1) // 2, idx)
        plt.title(f"Residual Graph {idx}")
        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="skyblue",
            node_size=500,
            font_weight="bold",
            arrows=True,
        )
    plt.tight_layout()
    plt.show()

# Function to find successors of a node in the graph
def successors(matrix, node):
    """
    Finds the successors of a given node in the graph represented by the adjacency matrix.

    Args:
    - matrix (list of list of int): Adjacency matrix representing the graph.
    - node (int): Node index to find successors for.

    Returns:
    - list of int: List of successor nodes.
    """
    return [j for j in range(len(matrix)) if matrix[node][j] != 0]

# Function to find predecessors of a node in the graph
def predecessors(matrix, node):
    """
    Finds the predecessors of a given node in the graph represented by the adjacency matrix.

    Args:
    - matrix (list of list of int): Adjacency matrix representing the graph.
    - node (int): Node index to find predecessors for.

    Returns:
    - list of int: List of predecessor nodes.
    """
    return [j for j in range(len(matrix)) if matrix[j][node] != 0]

# Function to check if the flow network is valid
def is_valid(matrix):
    """
    Checks if the flow network represented by the adjacency matrix is valid.

    Args:
    - matrix (list of list of int): Adjacency matrix representing the flow network.

    Returns:
    - tuple (bool, list, list): Tuple containing:
        - bool: True if the network is valid, False otherwise.
        - list: List of source nodes.
        - list: List of sink nodes.
    """
    list_1 = [
        node_index
        for node_index in range(len(matrix))
        if successors(matrix, node_index) == []
    ]
    list_2 = [
        node_index
        for node_index in range(len(matrix))
        if predecessors(matrix, node_index) == []
    ]
    for j in range(len(matrix)):
        if matrix[j][j] != 0:
            return False, [], []
    if len(list_1) == 1 and len(list_2) == 1:
        return True, list_2, list_1
    return False, [], []

# Function to find an augmenting path in the residual graph
def augmenting_path(residual_graph, source, sink):
    """
    Finds an augmenting path in the residual graph using BFS.

    Args:
    - residual_graph (list of list of int): Residual graph represented as adjacency matrix.
    - source (int): Source node index.
    - sink (int): Sink node index.

    Returns:
    - list of int or None: Augmenting path as a list of node indices or None if no path exists.
    """
    parent = [-1] * len(residual_graph)
    parent[source] = source
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v, capacity in enumerate(residual_graph[u]):
            if parent[v] == -1 and capacity > 0:
                parent[v] = u
                queue.append(v)
                if v == sink:
                    path = []
                    while v != source:
                        path.append(v)
                        v = parent[v]
                    path.append(source)
                    path.reverse()
                    return path
    return None

# Function to calculate the residual graph
def residual_graph(capacity_matrix, flow_matrix):
    """
    Calculates the residual graph based on the given capacity and flow matrices.

    Args:
    - capacity_matrix (list of list of int): Capacity matrix of the flow network.
    - flow_matrix (list of list of int): Flow matrix representing current flow values.

    Returns:
    - list of list of int: Residual graph as adjacency matrix.
    """
    n = len(capacity_matrix)
    residual = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            residual[i][j] = capacity_matrix[i][j] - flow_matrix[i][j]
    return residual

# Ford-Fulkerson algorithm to find maximum flow
def ford_fulkerson_algorithm(capacity_matrix, source, sink, flow_matrix):
    """
    Applies the Ford-Fulkerson algorithm to find the maximum flow in the flow network.

    Args:
    - capacity_matrix (list of list of int): Capacity matrix of the flow network.
    - source (int): Source node index.
    - sink (int): Sink node index.
    - flow_matrix (list of list of int): Flow matrix representing current flow values.

    Returns:
    - list of list of int: List of residual graphs encountered during the algorithm.
    """
    n = len(capacity_matrix)
    residual_graphs = []

    while True:
        residual = residual_graph(capacity_matrix, flow_matrix)
        augmenting_path_nodes = augmenting_path(residual, source, sink)

        if augmenting_path_nodes is None:
            break

        epsilon = min(
            residual[augmenting_path_nodes[i]][augmenting_path_nodes[i + 1]]
            for i in range(len(augmenting_path_nodes) - 1)
        )

        for i in range(len(augmenting_path_nodes) - 1):
            u, v = augmenting_path_nodes[i], augmenting_path_nodes[i + 1]
            flow_matrix[u][v] += epsilon
            flow_matrix[v][u] -= epsilon

        residual_graphs.append(residual)

    max_flow_value = sum(flow_matrix[source])
    print(f"Maximum flow value: {max_flow_value}")
    return residual_graphs

# Function to convert residual graph to NetworkX graph for visualization
def residual_to_networkx(residual):
    """
    Converts a residual graph (adjacency matrix) to a NetworkX graph for visualization.

    Args:
    - residual (list of list of int): Residual graph represented as adjacency matrix.

    Returns:
    - networkx.DiGraph: NetworkX directed graph object representing the residual graph.
    """
    G = nx.DiGraph()
    for i in range(len(residual)):
        for j in range(len(residual)):
            if residual[i][j] > 0:
                G.add_edge(i, j, capacity=residual[i][j])
    return G

# Main program
def main():
    file_path = "graph.txt"
    capacity_matrix = read_adjacency_matrix(file_path)
    valid, sources, sinks = is_valid(capacity_matrix)
    
    if valid:
        source = sources[0]
        sink = sinks[0]
        flow_matrix = np.zeros((len(capacity_matrix), len(capacity_matrix)), dtype=int)
        residual_graphs = ford_fulkerson_algorithm(
            capacity_matrix, source, sink, flow_matrix
        )
        show_graphs(residual_graphs)
    else:
        print("Invalid flow network.")

if __name__ == "__main__":
    main()
