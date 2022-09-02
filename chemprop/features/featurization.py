from typing import List, Tuple, Union
from itertools import zip_longest
import logging

import torch
import numpy as np

from math import log

from nodalpa.computational_graph import Instruction, TensorShape
from nodalpa.basic_computational_graph import BasicComputationalGraph

class Featurization_parameters:
    """
    A class holding molecule featurization parameters as attributes.
    """
    def __init__(self) -> None:

        # Node feature sizes
        self.MAX_SHAPE_DIMS = 4
        self.MAX_OP_ARGS = 5 # Maximum number of args and op is allowed
        self.OP_CODES = list(range(116))
        
        # TODO(kh): Consider adding something for dtype in TensorShape

        # Distance feature sizes ?
        self.PATH_DISTANCE_BINS = list(range(10))
        self.THREE_D_DISTANCE_MAX = 20
        self.THREE_D_DISTANCE_STEP = 1
        self.THREE_D_DISTANCE_BINS = list(range(0, self.THREE_D_DISTANCE_MAX + 1, self.THREE_D_DISTANCE_STEP))

        self.NODE_FDIM = len(self.OP_CODES) + self.MAX_SHAPE_DIMS
        self.EDGE_FDIM = self.MAX_OP_ARGS

# Create a global parameter object for reference throughout this module
PARAMS = Featurization_parameters()


def reset_featurization_parameters(logger: logging.Logger = None) -> None:
    """
    Function resets feature parameter values to defaults by replacing the parameters instance.
    """
    if logger is not None:
        debug = logger.debug
    else:
        debug = print
    debug('Setting molecule featurization parameters to default.')
    global PARAMS
    PARAMS = Featurization_parameters()


def get_node_fdim() -> int:
    """
    Gets the dimensionality of the node feature vector.
    :param overwrite_default_node: Whether to overwrite the default node descriptors
    :param is_reaction: Whether to add :code:`EXTRA_ATOM_FDIM` for reaction input when :code:`REACTION_MODE` is not None
    :return: The dimensionality of the node feature vector.
    """
    return PARAMS.NODE_FDIM


def get_edge_fdim(node_messages: bool = True) -> int:
    """
    Gets the dimensionality of the edge feature vector.
    :param node_messages: Whether node messages are being used. If node messages are used,
                          then the edge feature vector only contains edge features.
                          Otherwise it contains both node and edge features.
    :return: The dimensionality of the edge feature vector.
    """

    return PARAMS.BOND_FDIM + (not node_messages) * get_node_fdim()


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def log_shape(shape: TensorShape) -> List[float]:
    return [log(dim) for dim in TensorShape.dims()]


def node_features(inst: Instruction) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an node.
    :param node: An RDKit node.
    :param functional_groups: A k-hot vector indicating the functional groups the node belongs to.
    :return: A list containing the node features.
    """
    if inst is None:
        features = [0] * PARAMS.ATOM_FDIM
    else:
        features = onek_encoding_unk(int(inst.op_code()), PARAMS.OP_CODES) + \
            log_shape(inst.shape())
    return features


def edge_features(arg_pos: int) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a edge.
    :param edge: An RDKit edge.
    :return: A list containing the edge features.
    """
    return [arg_pos]


class ComputeGraph:
    """
    A :class:`ComputeGraph` represents the graph structure and featurization of a computational graph.
    A ComputeGraph computes the following attributes:
    * :code:`n_nodes`: The number of nodes in the graph.
    * :code:`n_edges`: The number of edges in the htaph.
    * :code:`f_nodes`: A mapping from an node index to a list of node features.
    * :code:`f_edges`: A mapping from a edge index to a list of edge features.
    * :code:`n2e`: A mapping from an node index to a list of incoming edge indices.
    * :code:`e2n`: A mapping from a edge index to the index of the node the edge originates from.
    * :code:`reverse_e2e`: A mapping from a edge index to the index of the reverse edge.
    """

    def __init__(self, graph: BasicComputationalGraph):
        """
        :param mol: A SMILES or an RDKit molecule.
        """

        self.n_nodes = 0  # number of nodes
        self.n_edges = 0  # number of edges
        self.f_nodes = []  # mapping from node index to node features
        self.f_edges = []  # mapping from edge index to concat(in_node, edge) features
        self.n2e = []  # mapping from node index to incoming edge indices
        self.e2n = []  # mapping from edge index to the index of the node the edge is coming from
        self.reverse_e2e = []  # mapping from edge index to the index of the reverse edge

        # Get node features
        self.f_nodes = [node_features(node) for node in graph.instructions()]

        self.n_nodes = len(self.f_nodes)

        # Initialize node to edge mapping for each node
        for _ in range(self.n_nodes):
            self.n2e.append([])

        # Get edge features
        for n1 in range(self.n_nodes):
            for arg in range(graph.instruction()[n1].operands()):
                edge = arg

                if edge is None:
                    continue

                f_edge = edge_features(edge)

                self.f_edges.append(self.f_nodes[a1] + f_edge)
                self.f_edges.append(self.f_nodes[a2] + f_edge)

                # Update index mappings
                # TODO(kh): Is this undirected?
                n2 = graph.instruction()[n1].operands()[arg].instruction_id()
                e = self.n_edges
                self.n2e[a2].append(b1)  # b1 = a1 --> a2
                self.e2n.append(a1)
                self.n2e[a1].append(b2)  # b2 = a2 --> a1
                self.e2n.append(a2)
                self.reverse_e2e.append(b2)
                self.reverse_e2e.append(b1)
                self.n_edges += 2

class BatchComputeGraph:
    """
    A :class:`BatchMolGraph` represents the graph structure and featurization of a batch of molecules.
    A BatchMolGraph contains the attributes of a :class:`MolGraph` plus:
    * :code:`node_fdim`: The dimensionality of the node feature vector.
    * :code:`edge_fdim`: The dimensionality of the edge feature vector (technically the combined node/edge features).
    * :code:`a_scope`: A list of tuples indicating the start and end node indices for each molecule.
    * :code:`b_scope`: A list of tuples indicating the start and end edge indices for each molecule.
    * :code:`max_num_edges`: The maximum number of edges neighboring an node in this batch.
    * :code:`e2e`: (Optional) A mapping from a edge index to incoming edge indices.
    * :code:`a2a`: (Optional): A mapping from an node index to neighboring node indices.
    """

    def __init__(self, graphs: List[ComputeGraph]):
        r"""
        :param mol_graphs: A list of :class:`MolGraph`\ s from which to construct the :class:`BatchMolGraph`.
        """
        self.node_fdim = get_node_fdim(overwrite_default_node=self.overwrite_default_node_features,
                                       is_reaction=self.is_reaction)
        self.edge_fdim = get_edge_fdim(overwrite_default_edge=self.overwrite_default_edge_features,
                                      overwrite_default_node=self.overwrite_default_node_features,
                                      is_reaction=self.is_reaction)

        # Start n_nodes and n_edges at 1 b/c zero padding
        self.n_nodes = 1  # number of nodes (start at 1 b/c need index 0 as padding)
        self.n_edges = 1  # number of edges (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_node_index, num_nodes) for each molecule
        self.b_scope = []  # list of tuples indicating (start_edge_index, num_edges) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_nodes = [[0] * self.node_fdim]  # node features
        f_edges = [[0] * self.edge_fdim]  # combined node/edge features
        n2e = [[]]  # mapping from node index to incoming edge indices
        e2n = [0]  # mapping from edge index to the index of the node the edge is coming from
        reverse_e2e = [0]  # mapping from edge index to the index of the reverse edge
        for graph in graphs:
            f_nodes.extend(graph.f_nodes)
            f_edges.extend(graph.f_edges)

            for n in range(graph.n_nodes):
                n2e.append([e + self.n_edges for e in graph.n2e[n]])

            for e in range(graph.n_edges):
                e2n.append(self.n_nodes + graph.e2n[e])
                reverse_e2e.append(self.n_edges + graph.reverse_e2e[e])

            self.n_scope.append((self.n_nodes, graph.n_nodes))
            self.e_scope.append((self.n_edges, graph.n_edges))
            self.n_nodes += graph.n_nodes
            self.n_edges += graph.n_edges

        self.max_num_edges = max(1, max(
            len(in_edges) for in_edges in n2e))  # max with 1 to fix a crash in rare case of all single-heavy-node mols

        self.f_nodes = torch.FloatTensor(f_nodes)
        self.f_edges = torch.FloatTensor(f_edges)
        self.n2e = torch.LongTensor([n2e[n] + [0] * (self.max_num_edges - len(n2e[n])) for n in range(self.n_nodes)])
        self.e2n = torch.LongTensor(e2n)
        self.reverse_e2e = torch.LongTensor(reverse_e2e)
        self.e2e = None  # try to avoid computing e2e b/c O(n_nodes^3)
        self.a2a = None  # only needed if using node messages

    def get_components(self, node_messages: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                                                   torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                                                   List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the :class:`BatchComputeGraph`.
        The returned components are, in order:
        * :code:`f_nodes`
        * :code:`f_edges`
        * :code:`n2e`
        * :code:`e2n`
        * :code:`reverse_e2e`
        * :code:`a_scope`
        * :code:`b_scope`
        :param node_messages: Whether to use node messages instead of edge messages. This changes the edge feature
                              vector to contain only edge features rather than both node and edge features.
        :return: A tuple containing PyTorch tensors with the node features, edge features, graph structure,
                 and scope of the nodes and edges (i.e., the indices of the molecules they belong to).
        """
        if node_messages:
            f_edges = self.f_edges[:, -get_edge_fdim(node_messages=node_messages,
                                                     overwrite_default_node=self.overwrite_default_node_features,
                                                     overwrite_default_edge=self.overwrite_default_edge_features):]
        else:
            f_edges = self.f_edges

        return self.f_nodes, f_edges, self.n2e, self.e2n, self.reverse_e2e, self.a_scope, self.b_scope

    def get_e2e(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each edge index to all the incoming edge indices.
        :return: A PyTorch tensor containing the mapping from each edge index to all the incoming edge indices.
        """
        if self.e2e is None:
            e2e = self.n2e[self.e2n]  # num_edges x max_num_edges
            # e2e includes reverse edge for each edge so need to mask out
            revmask = (e2e != self.reverse_e2e.unsqueeze(1).repeat(1, e2e.size(1))).long()  # num_edges x max_num_edges
            self.e2e = e2e * revmask

        return self.e2e

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each node index to all neighboring node indices.
        :return: A PyTorch tensor containing the mapping from each node index to all the neighboring node indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # n2e maps a2 to all incoming edges b
            # e2n maps each edge b to the node it comes from a1
            # thus e2n[n2e] maps node a2 to neighboring nodes a1
            self.a2a = self.e2n[self.n2e]  # num_nodes x max_num_edges

        return self.a2a


def graphs2batch(graphs: List[BasicComputationalGraph]) -> BatchComputeGraph:
    """
    Converts a list of SMILES or RDKit molecules to a :class:`BatchMolGraph` containing the batch of molecular graphs.
    :param mols: A list of SMILES or a list of RDKit molecules.
    :param node_features_batch: A list of 2D numpy array containing additional node features to featurize the molecule
    :param edge_features_batch: A list of 2D numpy array containing additional edge features to featurize the molecule
    :param overwrite_default_node_features: Boolean to overwrite default node descriptors by node_descriptors instead of concatenating
    :param overwrite_default_edge_features: Boolean to overwrite default edge descriptors by edge_descriptors instead of concatenating
    :return: A :class:`BatchMolGraph` containing the combined molecular graph for the molecules.
    """
    return BatchComputeGraph([ComputeGraph(graph)
                          for graph in graphs])
