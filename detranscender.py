import dgl.data.utils as dglutils
import dgl
import torch


def load_graph():
    glist, label_dict = dglutils.load_graphs("test/chr19_hifi30x.dgl")
    return glist, label_dict


def export_graph_to_csv(g, path):
    with open(path, "w") as f:
        for src, dst in zip(g.edges()[0].tolist(), g.edges()[1].tolist()):
            f.write(f"{src},{dst}\n")


def remove_transitive_edges(graph):
#    for n in graph.nodes():
#        for e in graph.out_edges():
    return 0


if __name__ == "__main__":
    print("Start:")

    #glist, label_dict = load_graph()
    #g = glist[0]

    g = dgl.graph((torch.tensor([1, 1, 1, 1, 2, 3, 4]), torch.tensor([2, 3, 4, 5, 3, 4, 5])))

    export_graph_to_csv(g, "out/before.csv")

    print("Loaded")
