from pymatgen.core.periodic_table import Element
from pymatgen.core.composition import Composition
from pymatgen.core.molecular_orbitals import MolecularOrbitals
from pymatgen.io.cif import CifParser

import numpy as np
import torch
from torch_geometric.data import Data

import networkx as nx
from matplotlib import pyplot as plt

def count_elements(compound:str) ->list:
    element_counts = {str(el): 0 for el in Element}
    composition = Composition(compound)
    for element, count in composition.items():
        element_counts[str(element)] += int(count)
    return list(element_counts.values())

def get_orbital_energy(compound:str) ->list:
    orbital = ['1s', '2s', '2p', '3s', '3p', '3d', '4s', '4p', '4d', '4f', '5s', '5p', '5d', '6s', '6p', '5f', '6d', '7s']
    orbital = {k:0.0 for k in orbital}
    aos = MolecularOrbitals(compound).aos_as_list()
    for orb in aos:
        orbital[orb[1]] = orb[2]
    return list(orbital.values())

def parse_cif(path) -> Data:
    parser = CifParser(path)
    sites = parser.get_structures()[0].as_dict()['sites']

    x = []
    edges_idx = [[],[]]
    edges_val = []
    pos = []

    for site in sites:
        specie = site['species'][0]['element']
        abc = site['abc']
        xyz = site['xyz']
        node_val = [*count_elements(specie), *get_orbital_energy(specie), *abc]
        node_pos = xyz
        x.append(node_val)
        pos.append(node_pos)

    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                edges_idx[0].append(i)
                edges_idx[1].append(j)
                edges_val.append(np.linalg.norm(np.array(pos[i])-np.array(pos[j])))

    x = torch.tensor(x, dtype=torch.float)
    edges_idx = torch.tensor(edges_idx, dtype=torch.long)
    edges_val = torch.tensor(edges_val, dtype=torch.float)
    pos = torch.tensor(pos, dtype=torch.float)

    graph = Data(x=x, edge_index=edges_idx, edge_attr=edges_val, pos=pos)
    return graph, sites

def vis(graph:Data, species:list=[], name:str="") -> None:
    G = nx.Graph()
    G.add_nodes_from(range(graph.num_nodes))
    edges = graph.edge_index.t().tolist()
    G.add_edges_from(edges)
    G_pos = {i:p.numpy() for i, p in enumerate(graph.pos)}
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    node_xyz = np.array([G_pos[v] for v in sorted(G)])
    edge_xyz = np.array([(G_pos[u], G_pos[v]) for u, v in G.edges()])
    ax.scatter(*node_xyz.T, s=100, ec="w")
    for i, label in enumerate(species):
        ax.text(*node_xyz[i], label)
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray", alpha=0.5)

    def _format_axes(ax):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(name)

    _format_axes(ax)
    fig.tight_layout()
    plt.show()
    return

if __name__ == '__main__':
    path = "datasets/cif/Nb2Zn2BiO8_mvc-2_computed.cif"
    graph, sites = parse_cif(path)
    vis(graph, [site['species'][0]['element'] for site in sites], "CIF")