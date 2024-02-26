import dgl.data.utils as dglutils
import os
import time


def export_graph_to_csv(g, path):
    with open(path, "w") as f:
        for e in zip(*g.edges()):
            f.write(f"{e[0].item()},{e[1].item()}\n")


def remove_transitive_edges(graph):
    candidates = [-1] * graph.number_of_nodes()
    marked_edges = []
    for n in graph.nodes():
        out_edges = graph.out_edges(n, form="all")
        for ei in range(out_edges[0].shape[0]):
            candidates[out_edges[1][ei].item()] = out_edges[2][ei].item()

        for ei in range(out_edges[0].shape[0]):
            out_edges2 = graph.out_edges(out_edges[1][ei].item(), form="all")
            for ei2 in range(out_edges2[0].shape[0]):
                if candidates[out_edges2[1][ei2].item()] != -1:
                    marked_edges.append(candidates[out_edges2[1][ei2].item()])

        candidates = [-1] * graph.number_of_nodes()

    graph.remove_edges(marked_edges)
    print(f"Removed: {len(marked_edges)} edges")


if __name__ == "__main__":

    input_path = "test/chr19_hifi30x.dgl"

    out_folder = "out"
    output_path = "out/chr19_hifi30x_optimised.dgl"

    before_csv_path = "out/before.csv"
    after_csv_path = "out/after.csv"

    print("Start:")

    isExist = os.path.exists(out_folder)
    if not isExist:
        os.makedirs(out_folder)

    glist, label_dict = dglutils.load_graphs(input_path)
    g = glist[0]

    # Test
    # g = dgl.graph((torch.tensor([1, 1, 1, 1, 2, 3, 4]), torch.tensor([2, 3, 4, 5, 3, 4, 5])))

    export_graph_to_csv(g, before_csv_path)

    start_time = time.time()
    remove_transitive_edges(g)
    end_time = time.time()

    dglutils.save_graphs(output_path, glist)
    export_graph_to_csv(g, after_csv_path)

    print(f"Finished: time={end_time-start_time} seconds")
