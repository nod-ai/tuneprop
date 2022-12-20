import threading
from collections import OrderedDict
from random import Random
from typing import Dict, Iterator, List, Optional, Union, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler

from .scaler import StandardScaler
from chemprop.features import get_features_generator
from chemprop.features import BatchComputeGraph, ComputeGraph

from chemprop.features.computational_graph.computational_graph import ComputationalGraph

def cache_graph() -> bool:
    r"""Returns whether RDKit compute_graphs will be cached."""
    return CACHE_MOL


def set_cache_graph(cache_graph: bool) -> None:
    r"""Sets whether RDKit compute_graphs will be cached."""
    global CACHE_MOL
    CACHE_MOL = cache_graph


class ComputeGraphDatapoint:
    """A :class:`ComputeGraphDatapoint` contains a single compute_graph and its associated features and targets."""

    def __init__(self,
                 graphs: List[ComputationalGraph],
                 targets: List[Optional[float]] = None,
                 row: OrderedDict = None,
                 data_weight: float = None,
                 gt_targets: List[bool] = None,
                 lt_targets: List[bool] = None,
                 features: np.ndarray = None,
                 node_features: np.ndarray = None,
                 node_descriptors: np.ndarray = None,
                 edge_features: np.ndarray = None,
                 overwrite_default_node_features: bool = False,
                 overwrite_default_edge_features: bool = False):
        """
        :param smiles: A list of the SMILES strings for the compute_graphs.
        :param targets: A list of targets for the compute_graph (contains None for unknown target values).
        :param row: The raw CSV row containing the information for this compute_graph.
        :param data_weight: Weighting of the datapoint for the loss function.
        :param gt_targets: Indicates whether the targets are an inequality regression target of the form ">x".
        :param lt_targets: Indicates whether the targets are an inequality regression target of the form "<x".
        :param features: A numpy array containing additional features (e.g., Morgan fingerprint).
        :param node_descriptors: A numpy array containing additional node descriptors to featurize the compute_graph
        :param edge_features: A numpy array containing additional edge features to featurize the compute_graph
        :param overwrite_default_node_features: Boolean to overwrite default node features by node_features
        :param overwrite_default_edge_features: Boolean to overwrite default edge features by edge_features
        """
        if features is not None and features_generator is not None:
            raise ValueError('Cannot provide both loaded features and a features generator.')

        self.compute_graphs = graphs
        self.targets = targets
        self.row = row
        self.features = features
        self.node_descriptors = node_descriptors
        self.node_features = node_features
        self.edge_features = edge_features
        self.overwrite_default_node_features = overwrite_default_node_features
        self.overwrite_default_edge_features = overwrite_default_edge_features

        if data_weight is not None:
            self.data_weight = data_weight
        if gt_targets is not None:
            self.gt_targets = gt_targets
        if lt_targets is not None:
            self.lt_targets = lt_targets

        # Fix nans in features
        replace_token = 0
        if self.features is not None:
            self.features = np.where(np.isnan(self.features), replace_token, self.features)

        # Fix nans in node_descriptors
        if self.node_descriptors is not None:
            self.node_descriptors = np.where(np.isnan(self.node_descriptors), replace_token, self.node_descriptors)

        # Fix nans in node_features
        if self.node_features is not None:
            self.node_features = np.where(np.isnan(self.node_features), replace_token, self.node_features)

        # Fix nans in edge_descriptors
        if self.edge_features is not None:
            self.edge_features = np.where(np.isnan(self.edge_features), replace_token, self.edge_features)

        # Save a copy of the raw features and targets to enable different scaling later on
        self.raw_features, self.raw_targets = self.features, self.targets
        self.raw_node_descriptors, self.raw_node_features, self.raw_edge_features = \
            self.node_descriptors, self.node_features, self.edge_features

    @property
    def graphs(self) -> List[ComputationalGraph]:
        """Gets the list of compute_graphs"""

        return self.compute_graphs

    @property
    def number_of_compute_graphs(self) -> int:
        """
        Gets the number of compute_graphs in the :class:`ComputeGraphDatapoint`.
        :return: The number of compute_graphs.
        """
        return len(self.compute_graphs)

    def set_graphs(self, graphs: List[ComputationalGraph]) -> None:
        self.compute_graphs = graphs

    def set_features(self, features: np.ndarray) -> None:
        """
        Sets the features of the compute_graph.
        :param features: A 1D numpy array of features for the compute_graph.
        """
        self.features = features

    def set_node_descriptors(self, node_descriptors: np.ndarray) -> None:
        """
        Sets the node descriptors of the compute_graph.
        :param node_descriptors: A 1D numpy array of features for the compute_graph.
        """
        self.node_descriptors = node_descriptors

    def set_node_features(self, node_features: np.ndarray) -> None:
        """
        Sets the node features of the compute_graph.
        :param node_features: A 1D numpy array of features for the compute_graph.
        """
        self.node_features = node_features

    def set_edge_features(self, edge_features: np.ndarray) -> None:
        """
        Sets the edge features of the compute_graph.
        :param edge_features: A 1D numpy array of features for the compute_graph.
        """
        self.edge_features = edge_features

    def extend_features(self, features: np.ndarray) -> None:
        """
        Extends the features of the compute_graph.
        :param features: A 1D numpy array of extra features for the compute_graph.
        """
        self.features = np.append(self.features, features) if self.features is not None else features

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.
        :return: The number of tasks.
        """
        return len(self.targets)

    def set_targets(self, targets: List[Optional[float]]):
        """
        Sets the targets of a compute_graph.
        :param targets: A list of floats containing the targets.
        """
        self.targets = targets

    def reset_features_and_targets(self) -> None:
        """Resets the features (node, edge, and compute_graph) and targets to their raw values."""
        self.features, self.targets = self.raw_features, self.raw_targets
        self.node_descriptors, self.node_features, self.edge_features = \
            self.raw_node_descriptors, self.raw_node_features, self.raw_edge_features


class ComputeGraphDataset(Dataset):
    r"""A :class:`ComputeGraphDataset` contains a list of :class:`ComputeGraphDatapoint`\ s with access to their attributes."""

    def __init__(self, data: List[ComputeGraphDatapoint]):
        r"""
        :param data: A list of :class:`ComputeGraphDatapoint`\ s.
        """
        self._data = data
        self._batch_graph = None
        self._random = Random()

    def graphs(self, flatten: bool = False) -> Union[List[ComputationalGraph], List[List[ComputationalGraph]]]:
        """
        Returns a list of the RDKit compute_graphs associated with each :class:`ComputeGraphDatapoint`.
        :param flatten: Whether to flatten the returned RDKit compute_graphs to a list instead of a list of lists.
        :return: A list of SMILES or a list of lists of RDKit compute_graphs, depending on :code:`flatten`.
        """
        if flatten:
            return [graph for d in self._data for graph in d.compute_graphs]

        return [d.compute_graphs for d in self._data]

    @property
    def number_of_compute_graphs(self) -> int:
        """
        Gets the number of compute_graphs in each :class:`ComputeGraphDatapoint`.
        :return: The number of compute_graphs.
        """
        return self._data[0].number_of_compute_graphs if len(self._data) > 0 else None

    def batch_graph(self) -> List[BatchComputeGraph]:
        r"""
        Constructs a :class:`~chemprop.features.BatchComputeGraph` with the graph featurization of all the compute_graphs.
        .. note::
           The :class:`~chemprop.features.BatchComputeGraph` is cached in after the first time it is computed
           and is simply accessed upon subsequent calls to :meth:`batch_graph`. This means that if the underlying
           set of :class:`ComputeGraphDatapoint`\ s changes, then the returned :class:`~chemprop.features.BatchComputeGraph`
           will be incorrect for the underlying data.
        :return: A list of :class:`~chemprop.features.BatchComputeGraph` containing the graph featurization of all the
                 compute_graphs in each :class:`ComputeGraphDatapoint`.
        """
        if self._batch_graph is None:
            self._batch_graph = []

            graphs = []
            for d in self._data:
                graphs_list = []
                for g in d.compute_graphs:
                    if len(d.compute_graphs) > 1 and (d.node_features is not None or d.edge_features is not None):
                        raise NotImplementedError('Atom descriptors are currently only supported with one compute_graph '
                                                    'per input (i.e., number_of_compute_graphs = 1).')

                    graph = ComputeGraph(g, d.node_features, d.edge_features,
                                            overwrite_default_node_features=d.overwrite_default_node_features,
                                            overwrite_default_edge_features=d.overwrite_default_edge_features)
                    graphs_list.append(graph)
                graphs.append(graphs_list)

            self._batch_graph = [BatchComputeGraph([g[i] for g in graphs]) for i in range(len(graphs[0]))]

        return self._batch_graph

    def features(self) -> List[np.ndarray]:
        """
        Returns the features associated with each compute_graph (if they exist).
        :return: A list of 1D numpy arrays containing the features for each compute_graph or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].features is None:
            return None

        return [d.features for d in self._data]

    def node_features(self) -> List[np.ndarray]:
        """
        Returns the node descriptors associated with each compute_graph (if they exit).
        :return: A list of 2D numpy arrays containing the node descriptors
                 for each compute_graph or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].node_features is None:
            return None

        return [d.node_features for d in self._data]

    def node_descriptors(self) -> List[np.ndarray]:
        """
        Returns the node descriptors associated with each compute_graph (if they exit).
        :return: A list of 2D numpy arrays containing the node descriptors
                 for each compute_graph or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].node_descriptors is None:
            return None

        return [d.node_descriptors for d in self._data]

    def edge_features(self) -> List[np.ndarray]:
        """
        Returns the edge features associated with each compute_graph (if they exit).
        :return: A list of 2D numpy arrays containing the edge features
                 for each compute_graph or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].edge_features is None:
            return None

        return [d.edge_features for d in self._data]

    def data_weights(self) -> List[float]:
        """
        Returns the loss weighting associated with each datapoint.
        """
        if not hasattr(self._data[0], 'data_weight'):
            return [1. for d in self._data]

        return [d.data_weight for d in self._data]

    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each compute_graph.
        :return: A list of lists of floats (or None) containing the targets.
        """
        return [d.targets for d in self._data]
    
    def mask(self) -> List[List[bool]]:
        """
        Returns whether the targets associated with each compute_graph and task are present.
        :return: A list of list of booleans associated with targets.
        """
        targets = self.targets()

        return [[t is not None for t in dt] for dt in targets]

    def gt_targets(self) -> List[np.ndarray]:
        """
        Returns indications of whether the targets associated with each compute_graph are greater-than inequalities.
        
        :return: A list of lists of booleans indicating whether the targets in those positions are greater-than inequality targets.
        """
        if not hasattr(self._data[0], 'gt_targets'):
            return None

        return [d.gt_targets for d in self._data]

    def lt_targets(self) -> List[np.ndarray]:
        """
        Returns indications of whether the targets associated with each compute_graph are less-than inequalities.
        
        :return: A list of lists of booleans indicating whether the targets in those positions are less-than inequality targets.
        """
        if not hasattr(self._data[0], 'lt_targets'):
            return None

        return [d.lt_targets for d in self._data]

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.
        :return: The number of tasks.
        """
        return self._data[0].num_tasks() if len(self._data) > 0 else None

    def features_size(self) -> int:
        """
        Returns the size of the additional features vector associated with the compute_graphs.
        :return: The size of the additional features vector.
        """
        return len(self._data[0].features) if len(self._data) > 0 and self._data[0].features is not None else None

    def node_descriptors_size(self) -> int:
        """
        Returns the size of custom additional node descriptors vector associated with the compute_graphs.
        :return: The size of the additional node descriptor vector.
        """
        return len(self._data[0].node_descriptors[0]) \
            if len(self._data) > 0 and self._data[0].node_descriptors is not None else None

    def node_features_size(self) -> int:
        """
        Returns the size of custom additional node features vector associated with the compute_graphs.
        :return: The size of the additional node feature vector.
        """
        return len(self._data[0].node_features[0]) \
            if len(self._data) > 0 and self._data[0].node_features is not None else None

    def edge_features_size(self) -> int:
        """
        Returns the size of custom additional edge features vector associated with the compute_graphs.
        :return: The size of the additional edge feature vector.
        """
        return len(self._data[0].edge_features[0]) \
            if len(self._data) > 0 and self._data[0].edge_features is not None else None

    def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0,
                           scale_node_descriptors: bool = False, scale_edge_features: bool = False) -> StandardScaler:
        """
        Normalizes the features of the dataset using a :class:`~chemprop.data.StandardScaler`.
        The :class:`~chemprop.data.StandardScaler` subtracts the mean and divides by the standard deviation
        for each feature independently.
        If a :class:`~chemprop.data.StandardScaler` is provided, it is used to perform the normalization.
        Otherwise, a :class:`~chemprop.data.StandardScaler` is first fit to the features in this dataset
        and is then used to perform the normalization.
        :param scaler: A fitted :class:`~chemprop.data.StandardScaler`. If it is provided it is used,
                       otherwise a new :class:`~chemprop.data.StandardScaler` is first fitted to this
                       data and is then used.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        :param scale_node_descriptors: If the features that need to be scaled are node features rather than compute_graph.
        :param scale_edge_features: If the features that need to be scaled are edge descriptors rather than compute_graph.
        :return: A fitted :class:`~chemprop.data.StandardScaler`. If a :class:`~chemprop.data.StandardScaler`
                 is provided as a parameter, this is the same :class:`~chemprop.data.StandardScaler`. Otherwise,
                 this is a new :class:`~chemprop.data.StandardScaler` that has been fit on this dataset.
        """
        if len(self._data) == 0 or \
                (self._data[0].features is None and not scale_edge_features and not scale_node_descriptors):
            return None

        if scaler is None:
            if scale_node_descriptors and not self._data[0].node_descriptors is None:
                features = np.vstack([d.raw_node_descriptors for d in self._data])
            elif scale_node_descriptors and not self._data[0].node_features is None:
                features = np.vstack([d.raw_node_features for d in self._data])
            elif scale_edge_features:
                features = np.vstack([d.raw_edge_features for d in self._data])
            else:
                features = np.vstack([d.raw_features for d in self._data])
            scaler = StandardScaler(replace_nan_token=replace_nan_token)
            scaler.fit(features)

        if scale_node_descriptors and not self._data[0].node_descriptors is None:
            for d in self._data:
                d.set_node_descriptors(scaler.transform(d.raw_node_descriptors))
        elif scale_node_descriptors and not self._data[0].node_features is None:
            for d in self._data:
                d.set_node_features(scaler.transform(d.raw_node_features))
        elif scale_edge_features:
            for d in self._data:
                d.set_edge_features(scaler.transform(d.raw_edge_features))
        else:
            for d in self._data:
                d.set_features(scaler.transform(d.raw_features.reshape(1, -1))[0])

        return scaler

    def normalize_targets(self) -> StandardScaler:
        """
        Normalizes the targets of the dataset using a :class:`~chemprop.data.StandardScaler`.
        The :class:`~chemprop.data.StandardScaler` subtracts the mean and divides by the standard deviation
        for each task independently.
        This should only be used for regression datasets.
        :return: A :class:`~chemprop.data.StandardScaler` fitted to the targets.
        """
        targets = [d.raw_targets for d in self._data]
        scaler = StandardScaler().fit(targets)
        scaled_targets = scaler.transform(targets).tolist()
        self.set_targets(scaled_targets)

        return scaler

    def set_targets(self, targets: List[List[Optional[float]]]) -> None:
        """
        Sets the targets for each compute_graph in the dataset. Assumes the targets are aligned with the datapoints.
        :param targets: A list of lists of floats (or None) containing targets for each compute_graph. This must be the
                        same length as the underlying dataset.
        """
        if not len(self._data) == len(targets):
            raise ValueError(
                "number of compute_graphs and targets must be of same length! "
                f"num compute_graphs: {len(self._data)}, num targets: {len(targets)}"
            )
        for i in range(len(self._data)):
            self._data[i].set_targets(targets[i])

    def reset_features_and_targets(self) -> None:
        """Resets the features (node, edge, and compute_graph) and targets to their raw values."""
        for d in self._data:
            d.reset_features_and_targets()

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e., the number of compute_graphs).
        :return: The length of the dataset.
        """
        return len(self._data)

    def __getitem__(self, item) -> Union[ComputeGraphDatapoint, List[ComputeGraphDatapoint]]:
        r"""
        Gets one or more :class:`ComputeGraphDatapoint`\ s via an index or slice.
        :param item: An index (int) or a slice object.
        :return: A :class:`ComputeGraphDatapoint` if an int is provided or a list of :class:`ComputeGraphDatapoint`\ s
                 if a slice is provided.
        """
        return self._data[item]


class ComputeGraphSampler(Sampler):
    """A :class:`ComputeGraphSampler` samples data from a :class:`ComputeGraphDataset` for a :class:`ComputeGraphDataLoader`."""

    def __init__(self,
                 dataset: ComputeGraphDataset,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative compute_graphs). Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if :code:`shuffle` is True.
        """
        super(Sampler, self).__init__()

        self.dataset = dataset
        self.class_balance = class_balance
        self.shuffle = shuffle

        self._random = Random(seed)

        if self.class_balance:
            indices = np.arange(len(dataset))
            has_active = np.array([any(target == 1 for target in datapoint.targets) for datapoint in dataset])

            self.positive_indices = indices[has_active].tolist()
            self.negative_indices = indices[~has_active].tolist()

            self.length = 2 * min(len(self.positive_indices), len(self.negative_indices))
        else:
            self.positive_indices = self.negative_indices = None

            self.length = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Creates an iterator over indices to sample."""
        if self.class_balance:
            if self.shuffle:
                self._random.shuffle(self.positive_indices)
                self._random.shuffle(self.negative_indices)

            indices = [index for pair in zip(self.positive_indices, self.negative_indices) for index in pair]
        else:
            indices = list(range(len(self.dataset)))

            if self.shuffle:
                self._random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """Returns the number of indices that will be sampled."""
        return self.length


def construct_compute_graph_batch(data: List[ComputeGraphDatapoint]) -> ComputeGraphDataset:
    r"""
    Constructs a :class:`ComputeGraphDataset` from a list of :class:`ComputeGraphDatapoint`\ s.
    Additionally, precomputes the :class:`~chemprop.features.BatchComputeGraph` for the constructed
    :class:`ComputeGraphDataset`.
    :param data: A list of :class:`ComputeGraphDatapoint`\ s.
    :return: A :class:`ComputeGraphDataset` containing all the :class:`ComputeGraphDatapoint`\ s.
    """
    data = ComputeGraphDataset(data)
    data.batch_graph()  # Forces computation and caching of the BatchComputeGraph for the compute_graphs

    return data


class ComputeGraphDataLoader(DataLoader):
    """A :class:`ComputeGraphDataLoader` is a PyTorch :class:`DataLoader` for loading a :class:`ComputeGraphDataset`."""

    def __init__(self,
                 dataset: ComputeGraphDataset,
                 batch_size: int = 50,
                 num_workers: int = 8,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param dataset: The :class:`ComputeGraphDataset` containing the compute_graphs to load.
        :param batch_size: Batch size.
        :param num_workers: Number of workers used to build batches.
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative compute_graphs). Class balance is only available for single task
                              classification datasets. Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if shuffle is True.
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._class_balance = class_balance
        self._shuffle = shuffle
        self._seed = seed
        self._context = None
        self._timeout = 0
        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread and self._num_workers > 0:
            self._context = 'forkserver'  # In order to prevent a hanging
            self._timeout = 3600  # Just for sure that the DataLoader won't hang

        self._sampler = ComputeGraphSampler(
            dataset=self._dataset,
            class_balance=self._class_balance,
            shuffle=self._shuffle,
            seed=self._seed
        )

        super(ComputeGraphDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=construct_compute_graph_batch,
            multiprocessing_context=self._context,
            timeout=self._timeout
        )

    @property
    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each compute_graph.
        :return: A list of lists of floats (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')

        return [self._dataset[index].targets for index in self._sampler]

    @property
    def gt_targets(self) -> List[List[Optional[bool]]]:
        """
        Returns booleans for whether each target is an inequality rather than a value target, associated with each compute_graph.
        :return: A list of lists of booleans (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')
        
        if not hasattr(self._dataset[0],'gt_targets'):
            return None

        return [self._dataset[index].gt_targets for index in self._sampler]

    @property
    def lt_targets(self) -> List[List[Optional[bool]]]:
        """
        Returns booleans for whether each target is an inequality rather than a value target, associated with each compute_graph.
        :return: A list of lists of booleans (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')

        if not hasattr(self._dataset[0],'lt_targets'):
            return None

        return [self._dataset[index].lt_targets for index in self._sampler]


    @property
    def iter_size(self) -> int:
        """Returns the number of data points included in each full iteration through the :class:`ComputeGraphDataLoader`."""
        return len(self._sampler)

    def __iter__(self) -> Iterator[ComputeGraphDataset]:
        r"""Creates an iterator which returns :class:`ComputeGraphDataset`\ s"""
        return super(ComputeGraphDataLoader, self).__iter__()

    
def make_graphs(smiles: List[str]):
    """
    Builds a list of compute_graphs (or a list of tuples of compute_graphs if reaction is True) for a list of smiles.
    :param smiles: List of SMILES strings.
    :return: List of compute_graphs or list of tuple of compute_graphs.
    """
    graphs = []
    for s, keep_h, add_h in zip(smiles):
        graphs.append(SMILES_TO_MOL[s] if s in SMILES_TO_MOL else make_graph(s, keep_h, add_h))
    return graphs
