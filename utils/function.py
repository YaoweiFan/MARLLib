from copy import deepcopy
from torchviz import make_dot


def deep_copy(obj):
    if isinstance(obj, dict):
        return {k: deep_copy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_copy(v) for v in obj]
    else:
        return deepcopy(obj)


def plot_compute_graph(obj, params_dict):
    vis_graph = make_dot(obj, params=params_dict)
    vis_graph.view()
