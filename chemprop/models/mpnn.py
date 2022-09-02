from typing import List, Union, Tuple
from functools import reduce

import numpy as np
import torch
import torch.nn as nn

from chemprop.args import TrainArgs
from chemprop.features import BatchComputeGraph, get_node_fdim, get_edge_fdim, graphs2batch
from chemprop.nn_utils import index_select_ND

from nodalpa.computational_graph import ComputationalGraph


class MPNNEncoder(nn.Module):
    """An :class:`MPNNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(self, args: TrainArgs, node_fdim: int, edge_fdim: int, hidden_size: int = None,
                 bias: bool = None, depth: int = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param node_fdim: Atom feature vector dimension.
        :param edge_fdim: Bond feature vector dimension.
        :param hidden_size: Hidden layers dimension
        :param bias: Whether to add bias to linear layers
        :param depth: Number of message passing steps
       """
        super(MPNNEncoder, self).__init__()
        self.node_fdim = node_fdim # opeartor_features: operation_id, shape_i...
        self.edge_fdim = edge_fdim #edge_features: null? index?
        self.node_messages = args.node_messages # centers messages on nodes
        self.hidden_size = hidden_size or args.hidden_size #? = atoms?
        self.bias = bias or args.bias #?
        self.depth = depth or args.depth # w/e
        self.dropout = args.dropout #?
        self.layers_per_message = 1 # w/e
        self.undirected = args.undirected # rm, always directed
        self.device = args.device #?
        self.aggregation = args.aggregation #?
        self.aggregation_norm = args.aggregation_norm #?

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation) #kh relu?

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.node_fdim if self.node_messages else self.edge_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.node_messages:
            w_h_input_size = self.hidden_size + self.edge_fdim #why?
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.node_fdim + self.hidden_size, self.hidden_size)

        # layer after concatenating the descriptors if args.node_descriptors == descriptors
        if args.node_descriptors == 'descriptor':
            self.node_descriptors_size = args.node_descriptors_size
            self.node_descriptors_layer = nn.Linear(self.hidden_size + self.node_descriptors_size,
                                                    self.hidden_size + self.node_descriptors_size,)

    def forward(self,
                graph: BatchComputeGraph) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.
        :param mol_graph: A :class:`~chemprop.features.featurization.BatchMolGraph` representing
                          a batch of molecular graphs.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """

        f_nodes, f_edges, n2e, e2n, e2reversee, n_scope, e_scope = mol_graph.get_components(node_messages=self.node_messages)
        f_nodes, f_edges, n2e, e2n, e2reversee = f_nodes.to(self.device), f_edges.to(self.device), n2e.to(self.device), e2n.to(self.device), e2reversee.to(self.device)

        if self.node_messages:
            n2n = graph.get_n2n().to(self.device)

        # Input
        if self.node_messages:
            input = self.W_i(f_nodes)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_edges)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            # if self.undirected:
            #     message = (message + message[e2reversee]) / 2

            if self.node_messages:
                # m(v, t+1)
                nei_n_message = index_select_ND(message, n2n)  # num_atoms x max_num_bonds x hidden
                nei_f_edges = index_select_ND(f_edges, n2e)  # num_atoms x max_num_bonds x edge_fdim
                nei_message = torch.cat((nei_n_message, nei_f_edges), dim=2)  # num_atoms x max_num_bonds x hidden + edge_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + edge_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      n_message = sum(nei_n_message)      rev_message
                nei_n_message = index_select_ND(message, n2e)  # num_atoms x max_num_bonds x hidden
                n_message = nei_n_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[e2reversee]  # num_bonds x hidden
                message = n_message[e2n] - rev_message  # num_bonds x hidden

            # U_t
            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        # One last step with dropout and a separate network layer
        n2x = n2n if self.node_messages else n2e
        nei_n_message = index_select_ND(message, n2x)  # num_atoms x max_num_bonds x hidden
        n_message = nei_n_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_nodes, n_message], dim=1)  # num_atoms x (node_fdim + hidden)
        node_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        node_hiddens = self.dropout_layer(node_hiddens)  # num_atoms x hidden

        # concatenate the atom descriptors
        # probably do not use, most equivalent would be having a separate file for op hidden values
        if node_descriptors_batch is not None:
            if len(node_hiddens) != len(node_descriptors_batch):
                raise ValueError(f'The number of atoms is different from the length of the extra atom features')

            node_hiddens = torch.cat([node_hiddens, node_descriptors_batch], dim=1)     # num_atoms x (hidden + descriptor size)
            node_hiddens = self.node_descriptors_layer(node_hiddens)                    # num_atoms x (hidden + descriptor size)
            node_hiddens = self.dropout_layer(node_hiddens)                             # num_atoms x (hidden + descriptor size)

        # Readout
        hidden_vecs = []
        for i, (n_start, n_size) in enumerate(n_scope):
            if n_size == 0:
                hidden_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = node_hiddens.narrow(0, n_start, n_size)
                hidden_vec = cur_hiddens  # (num_atoms, hidden_size)
                if self.aggregation == 'mean':
                    hidden_vec = hidden_vec.sum(dim=0) / n_size
                elif self.aggregation == 'sum':
                    hidden_vec = hidden_vec.sum(dim=0)
                elif self.aggregation == 'norm':
                    hidden_vec = hidden_vec.sum(dim=0) / self.aggregation_norm
                hidden_vecs.append(hidden_vec)

        hidden_vecs = torch.stack(hidden_vecs, dim=0)  # (num_molecules, hidden_size)

        return hidden_vecs  # num_molecules x hidden


class MPNN(nn.Module):
    """An :class:`MPNN` is a wrapper around :class:`MPNNEncoder` which featurizes input as needed."""

    def __init__(self,
                 args: TrainArgs,
                 node_fdim: int = None,
                 edge_fdim: int = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param node_fdim: Atom feature vector dimension.
        :param edge_fdim: Bond feature vector dimension.
        """
        super(MPNN, self).__init__()
        self.reaction = args.reaction
        self.reaction_solvent = args.reaction_solvent
        self.node_fdim = node_fdim or get_node_fdim(overwrite_default_atom=args.overwrite_default_node_features,
                                                     is_reaction=(self.reaction or self.reaction_solvent))
        self.edge_fdim = edge_fdim or get_edge_fdim(overwrite_default_atom=args.overwrite_default_node_features,
                                                    overwrite_default_bond=args.overwrite_default_edge_features,
                                                    node_messages=args.node_messages,
                                                    is_reaction=(self.reaction or self.reaction_solvent))
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.device = args.device
        self.node_descriptors = args.node_descriptors
        self.overwrite_default_node_features = args.overwrite_default_node_features
        self.overwrite_default_edge_features = args.overwrite_default_edge_features

        if self.features_only:
            return

        if not self.reaction_solvent:
            if args.MPNN_shared:
                self.encoder = nn.ModuleList([MPNNEncoder(args, self.node_fdim, self.edge_fdim)] * args.number_of_molecules)
            else:
                self.encoder = nn.ModuleList([MPNNEncoder(args, self.node_fdim, self.edge_fdim)
                                               for _ in range(args.number_of_molecules)])
        else:
            self.encoder = MPNNEncoder(args, self.node_fdim, self.edge_fdim)
            # Set separate node_fdim and edge_fdim for solvent molecules
            self.node_fdim_solvent = get_node_fdim(overwrite_default_atom=args.overwrite_default_node_features,
                                                   is_reaction=False)
            self.edge_fdim_solvent = get_edge_fdim(overwrite_default_atom=args.overwrite_default_node_features,
                                                   overwrite_default_bond=args.overwrite_default_edge_features,
                                                   node_messages=args.node_messages,
                                                   is_reaction=False)
            self.encoder_solvent = MPNNEncoder(args, self.node_fdim_solvent, self.edge_fdim_solvent,
                                               args.hidden_size_solvent, args.bias_solvent, args.depth_solvent)

    def forward(self, batch: List[List[ComputationalGraph]]) -> torch.FloatTensor:
        """
        Encodes a batch of molecules.
        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        if type(batch[0]) != BatchMolGraph:
            # Group first molecules, second molecules, etc for mol2graph
            batch = [[mols[i] for mols in batch] for i in range(len(batch[0]))]

            # TODO: handle node_descriptors_batch with multiple molecules per input
            if self.node_descriptors == 'feature':
                if len(batch) > 1:
                    raise NotImplementedError('Atom/bond descriptors are currently only supported with one molecule '
                                              'per input (i.e., number_of_molecules = 1).')

                batch = [
                    mol2graph(
                        mols=b,
                        node_features_batch=node_features_batch,
                        edge_features_batch=edge_features_batch,
                        overwrite_default_node_features=self.overwrite_default_node_features,
                        overwrite_default_edge_features=self.overwrite_default_edge_features
                    )
                    for b in batch
                ]
                batch = [mol2graph(b) for b in batch]

        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float().to(self.device)

            if self.features_only:
                return features_batch

        if self.node_descriptors == 'descriptor':
            if len(batch) > 1:
                raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                          'per input (i.e., number_of_molecules = 1).')

            encodings = [enc(ba, node_descriptors_batch) for enc, ba in zip(self.encoder, batch)]
        else:
            if not self.reaction_solvent:
                 encodings = [enc(ba) for enc, ba in zip(self.encoder, batch)]
            else:
                 encodings = []
                 for ba in batch:
                     if ba.is_reaction:
                         encodings.append(self.encoder(ba))
                     else:
                         encodings.append(self.encoder_solvent(ba))

        output = reduce(lambda x, y: torch.cat((x, y), dim=1), encodings)

        if self.use_input_features:
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view(1, -1)

            output = torch.cat([output, features_batch], dim=1)

        return 
