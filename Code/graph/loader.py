# define the functions used for later 
from torch_geometric.loader import ImbalancedSampler, NeighborLoader
from torch.utils.data import IterableDataset
import torch
import numpy as np
import torch
from  utilities.bat import * 
import copy 


class BatLoader:
    def __init__(self, data, include_mask = True, warmup = 10):
        # process the data 
        self.ori_data = data 
        self.aug_data = copy.deepcopy(data)
        if include_mask:
            if not hasattr(data, 'train_mask'):
                self.aug_data.train_mask = data.mask 
        else: # include all nodes 
            self.aug_data.train_mask = torch.ones(self.aug_data.x.shape[0]).bool()
        self.augmenter = BatAugmenter().init_with_data(self.aug_data) 
        self.warmup = warmup
        self.graph_property = 'augment'
     
    def augment(self, model, epoch):
        if epoch <= self.warmup: # return the original graph
            return self.ori_data
        else:
            self.aug_data.x, self.aug_data.edge_index, self.aug_data.edge_attr, info = self.augmenter.augment(model, self.ori_data.x, self.ori_data.edge_index, self.ori_data.edge_attr)
            self.aug_data.y, self.aug_data.train_mask = self.augmenter.adapt_labels_and_train_mask(self.ori_data.y, self.ori_data.mask)
            self.aug_data.mask = self.aug_data.train_mask
            return self.aug_data


class CachedGraphLoader:
    def __init__(self, data, batch_size, k_hops, num_sample=-1, input_nodes=None, by=None, random_state =None):
        """
        Initializes the graph loader.
        
        :param data: PyG data object
        :param batch_size: Batch size for training
        :param k_hops: Number of hops for neighborhood sampling
        :param num_sample: Number of neighbors to sample at each hop (-1 means all)
        :param input_nodes: Optional subset of input nodes to sample from
        :param strategy: Sampling strategy (None or 'weights')
        """
        self.data = data
        self.batch_size = batch_size
        self.k_hops = k_hops
        self.num_sample = num_sample
        self.input_nodes = input_nodes
        self.by = by
        self.sampler = None
        self.graph_property = 'batch'
        self.epoch_count =0 
        self.cached_subgraphs = []  # ⬅️ pre-sampled subgraphs
        # build first cache
        self.random_state = random_state
        self._refresh_cache()

    def _refresh_cache(self):
        """Resample subgraphs and store them in memory."""
        print(f"[INFO] Refreshing subgraph cache (epoch={self.epoch_count})...")
        del self.cached_subgraphs # delect memory
        self.cached_subgraphs = []
    
        neighbors = [self.num_sample for _ in range(self.k_hops)]
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        if self.by == 'weights':
            self.sampler = ImbalancedSampler(self.data.y.detach().clone().long())
            loader = NeighborLoader(
                self.data,
                batch_size=self.batch_size,
                num_neighbors=neighbors,
                shuffle=False,
                sampler=self.sampler,
                input_nodes=self.input_nodes
            )
        else:
            loader = NeighborLoader(
                self.data,
                batch_size=self.batch_size,
                num_neighbors=neighbors,
                shuffle=True,
                input_nodes=self.input_nodes
            )
        for subg in loader:
            self.cached_subgraphs.append(copy.deepcopy(subg))

    def __iter__(self):
        for subg in self.cached_subgraphs:
            yield subg




class CachedSubgraphLoader(IterableDataset):
    def __init__(
        self,
        data,
        k_hops: int = 3,
        batch_size: int = 64,
        by: str = None,
        num_sample: int = -1,
        repeat_pos: int = 1,
        group_size: int = 2,
        random_state: int = None,
        **loader_kwargs
    ):
        self.data = data
        self.k_hops = k_hops
        self.num_sample = num_sample
        self.batch_size = batch_size
        self.by = by
        self.repeat_pos = repeat_pos
        self.group_size = group_size
        self.random_state = random_state
        self.loader_kwargs = loader_kwargs
        self.epoch_count = 0
        self.graph_property = 'batch'
        self.cached_subgraphs = []  # ⬅️ pre-sampled subgraphs
        self.sampler = None

        # build first cache
        self._refresh_cache()

    def _create_input_nodes(self) -> torch.Tensor:
        mask = getattr(self.data, "mask", None)
        if mask is None:
            raise ValueError("Data must have a 'mask' attribute for selecting input nodes.")

        node_ids = np.where(mask)[0]
        node_labels = np.asarray(self.data.y)[mask]
        rng = np.random.default_rng(self.random_state)

        if self.by is None:
            return torch.tensor(rng.permutation(node_ids), dtype=torch.long)

        elif self.by == 'weights':
            c0 = node_ids[node_labels == 0]
            c1 = node_ids[node_labels == 1]
            rng.shuffle(c0); rng.shuffle(c1)
            half = self.batch_size // 2
            total_batches = min(len(c0), len(c1)) // half
            batches = [
                rng.permutation(np.concatenate([c0[i*half:(i+1)*half],
                                                c1[i*half:(i+1)*half]]))
                for i in range(total_batches)
            ]
            return torch.tensor(np.concatenate(batches), dtype=torch.long)

        elif self.by == 'qids':
            c0 = node_ids[node_labels == 0]
            c1 = node_ids[node_labels == 1]
            pos = np.repeat(c1, self.repeat_pos)
            batches = [
                rng.permutation(np.concatenate([[p], rng.choice(c0, size=self.group_size - 1, replace=False)]))
                for p in pos
            ]
            return torch.tensor(np.concatenate(batches), dtype=torch.long)
        else:
            raise ValueError("Invalid 'by' parameter")

    def _refresh_cache(self):
        """Resample subgraphs and store them in memory."""
        print(f"[INFO] Refreshing subgraph cache (epoch={self.epoch_count})...")
        del self.cached_subgraphs # delect memory
        self.cached_subgraphs = []
        input_nodes = self._create_input_nodes()
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        loader = NeighborLoader(
            self.data,
            num_neighbors=[self.num_sample] * self.k_hops,
            batch_size=self.batch_size if self.by != 'qids' else self.batch_size * self.group_size,
            shuffle=False,
            input_nodes=input_nodes,
            disjoint=True,
            pin_memory=True,
            **self.loader_kwargs
        )

        for subg in loader:
            input_id = input_nodes[subg.input_id]
            subg.y = self.data.y[input_id]
            subg.centroid_feats = self.data.x[input_id, :]
            subg.mask = self.data.mask[input_id]
            subg.group_size = self.group_size
            self.cached_subgraphs.append(copy.deepcopy(subg))

    def __iter__(self):
        for subg in self.cached_subgraphs:
            yield subg
