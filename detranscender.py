import dgl
from dgl.data.utils import load_graphs

def load_graph():
	glist, label_dict = load_graphs("./data.bin") # glist will be [g1, g2]

print("Start:")
