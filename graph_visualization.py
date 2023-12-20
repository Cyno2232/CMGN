import networkx as nx
import matplotlib.pyplot as plt


def graph_visualization(g):
    g_nx = g.to_networkx()
    pos = nx.circular_layout(g_nx)
    nx.draw(g_nx, pos, with_labels=True, node_color='lightblue', edge_color='gray', arrows=True, width=0.5)

    plt.show()

