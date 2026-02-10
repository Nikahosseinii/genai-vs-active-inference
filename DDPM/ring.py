import networkx as nx
import matplotlib.pyplot as plt
import torch

def create_ring_graph(n):
    # Create an empty graph
    G = nx.Graph()

    # Add nodes to the graph
    G.add_nodes_from(range(n))

    # Add edges to connect each node to its four nearest neighbors
    for i in range(n):
        G.add_edge(i, (i+1) % n)
        G.add_edge(i, (i-1) % n)
        G.add_edge(i, (i+2) % n)
        G.add_edge(i, (i-2) % n)

    return G

# Create a ring graph with 64 nodes
graph = create_ring_graph(64)

#save graph as tensor
adjacency_matrix = nx.to_numpy_matrix(graph)
adjacency_tensor = torch.Tensor(adjacency_matrix)

file_path="/Users/nikahosseini/Desktop/trained_data/ring/circular.pt"

# Save the tensor to a .pt file
torch.save(adjacency_tensor, file_path)

# Draw the graph
pos = nx.circular_layout(graph)
nx.draw(graph, pos, with_labels=True, node_color='gray', node_size=5, edge_color='gray', width=1.5)

# Display the graph
plt.axis('equal')
plt.show()
