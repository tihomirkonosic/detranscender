import os
import time

import dgl
import dgl.data.utils as dgl_utils
import torch


def is_comparable(g, edge_a, edge_b, edge_c):
    eps = 0.12
    overlap_lengths = g.edata["overlap_length"]
    read_lengths = g.ndata["read_length"]

    node_a = g.edges()[0][edge_a]
    node_b = g.edges()[0][edge_b]
    length_a = read_lengths[node_a].item() - overlap_lengths[edge_a].item()
    length_b = read_lengths[node_b].item() - overlap_lengths[edge_b].item()
    length_c = read_lengths[node_a].item() - overlap_lengths[edge_c].item()
    a = length_a + length_b
    b = length_c

    return (b * (1 - eps) <= a <= b * (1 + eps)) or (a * (1 - eps) <= b <= a * (1 + eps))


def remove_transitive_edges(g):
    candidates = [-1] * g.number_of_nodes()
    marked_edges = []
    for n in g.nodes():
        out_edges = g.out_edges(n, form="all")
        for ei in range(out_edges[0].shape[0]):
            candidates[out_edges[1][ei].item()] = out_edges[2][ei].item()

        for ei in range(out_edges[0].shape[0]):
            out_edges2 = g.out_edges(out_edges[1][ei].item(), form="all")
            for ei2 in range(out_edges2[0].shape[0]):
                if (candidates[out_edges2[1][ei2].item()] != -1 and
                        is_comparable(g,
                                      out_edges[2][ei].item(),
                                      out_edges2[2][ei2].item(),
                                      candidates[out_edges2[1][ei2].item()])):
                    marked_edges.append(candidates[out_edges2[1][ei2].item()])

        candidates = [-1] * g.number_of_nodes()

    edges_before = g.number_of_edges()
    g.remove_edges(marked_edges)
    print(f"Removed transitive edges; after: {g.number_of_edges()}; before: {edges_before}")


def load_graph(path):
    glist, label_dict = dgl_utils.load_graphs(path)
    return glist[0]


def export_to_csv(g, path):
    with open(path, "w") as f:
        for e in zip(*g.edges()):
            f.write(f"{e[0].item()},{e[1].item()}\n")


def export_to_gfa(g, path, one_strand=True):
    with open(path, "w") as f:
        for n in g.nodes():
            if one_strand and g.ndata['read_strand'][n.item()].item() != -1:
                continue
            f.write(f"S\tN_{n.item()}\t*\tLN:i:{g.ndata['read_length'][n.item()].item()}\n")

        edges = g.edges(form="all")
        for ei in range(edges[0].shape[0]):
            if one_strand and g.ndata['read_strand'][edges[0][ei].item()].item() != -1:
                continue
            f.write(f"L\tN_{edges[0][ei].item()}\t+\tN_{edges[1][ei].item()}\t+\t{g.edata['overlap_length'][ei].item()}M\n")


def generate_test_graph():
    g = dgl.graph((torch.tensor([0, 0, 0, 0, 1, 2, 3]), torch.tensor([1, 2, 3, 4, 2, 3, 4])))
    g.edata["overlap_length"] = torch.tensor([100, 200, 700, 800, 100, 100, 100])
    return g


if __name__ == "__main__":

    input_path = "test/chr19.dgl"

    out_folder = "out"
    output_dgl_file = "chr19_hifi30x_optimised.dgl"

    before_gfa_file = "before.gfa"
    after_gfa_file = "after.gfa"

    print("Start:")

    isExist = os.path.exists(out_folder)
    if not isExist:
        os.makedirs(out_folder)

    graph = load_graph(input_path)
    # graph = generate_test_graph()

    # export_to_csv(graph, os.path.join(out_folder, before_csv_file))
    export_to_gfa(graph, os.path.join(out_folder, before_gfa_file), one_strand=False)

    print("Graph loaded")

    start_time = time.time()
    remove_transitive_edges(graph)
    end_time = time.time()

    print(f"Removed transitive edges in {end_time - start_time} seconds")

    dgl_utils.save_graphs(os.path.join(out_folder, output_dgl_file), [graph])
    # export_to_csv(graph, os.path.join(out_folder, after_csv_file))
    export_to_gfa(graph, os.path.join(out_folder, after_gfa_file), one_strand=False)

    print("Done")
