import os
import numpy as np
import scipy.sparse as sp 
import torch
from torch.utils.data import Dataset, DataLoader


def write_log(content, log_path):
    print(content)
    if log_path is not None:
        log = open(log_path, "a")
        log.write(f"{content}\n")
        log.close()


def print_statistics(X, string, log_path):
    write_log('-'*10 + string + '-'*10, log_path)
    write_log(f'Avg non-zeros in row:    {X.sum(1).mean(0).item():8.4f}', log_path)
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    write_log(f'Ratio of non-empty rows: {len(unique_nonzero_row_indice)/X.shape[0]:8.4f}', log_path)
    write_log(f'Ratio of non-empty cols: {len(unique_nonzero_col_indice)/X.shape[1]:8.4f}', log_path)
    write_log(f'Density of matrix:       {len(nonzero_row_indice)/(X.shape[0]*X.shape[1]):8.4f}', log_path)


def load_external_embedding_tensor(path, num_expected, entity_name):
    if not path:
        raise ValueError(f"{entity_name} embedding path is required")

    payload = torch.load(path, map_location="cpu")
    if torch.is_tensor(payload):
        embedding = payload.detach().float()
    elif isinstance(payload, np.ndarray):
        embedding = torch.from_numpy(payload).float()
    elif isinstance(payload, dict):
        rows = []
        for idx in range(num_expected):
            if idx in payload:
                value = payload[idx]
            elif str(idx) in payload:
                value = payload[str(idx)]
            else:
                raise KeyError(
                    f"{entity_name} embedding file {path} is missing id {idx}; "
                    "expected a full mapping aligned with current dataset ids"
                )
            value = torch.as_tensor(value).detach().float().view(-1)
            rows.append(value)
        embedding = torch.stack(rows, dim=0)
    else:
        raise TypeError(
            f"Unsupported payload type {type(payload)} in {path}; "
            "expected a torch tensor, numpy array, or id->embedding dict"
        )

    if embedding.ndim != 2:
        raise ValueError(f"{entity_name} embedding tensor from {path} must be 2D")
    if embedding.shape[0] != num_expected:
        raise ValueError(
            f"{entity_name} embedding tensor from {path} has {embedding.shape[0]} rows, "
            f"expected {num_expected}"
        )
    return embedding.contiguous()


def build_weighted_cbdm_input_graph(u_b_graph, user_embeddings, bundle_embeddings, gamma, eps):
    if not sp.isspmatrix_csr(u_b_graph):
        u_b_graph = u_b_graph.tocsr()

    if user_embeddings.ndim != 2 or bundle_embeddings.ndim != 2:
        raise ValueError("External user and bundle embeddings must be 2D")
    if user_embeddings.shape[1] != bundle_embeddings.shape[1]:
        raise ValueError("User and bundle embeddings must share the same embedding dimension")
    if user_embeddings.shape[0] != u_b_graph.shape[0]:
        raise ValueError(
            f"user embedding rows ({user_embeddings.shape[0]}) do not match graph users ({u_b_graph.shape[0]})"
        )
    if bundle_embeddings.shape[0] != u_b_graph.shape[1]:
        raise ValueError(
            f"bundle embedding rows ({bundle_embeddings.shape[0]}) do not match graph bundles ({u_b_graph.shape[1]})"
        )

    user_embeddings = np.asarray(user_embeddings, dtype=np.float32)
    bundle_embeddings = np.asarray(bundle_embeddings, dtype=np.float32)

    indptr = u_b_graph.indptr
    indices = u_b_graph.indices
    weighted_data = np.empty_like(u_b_graph.data, dtype=np.float32)

    for user_id in range(u_b_graph.shape[0]):
        start = indptr[user_id]
        end = indptr[user_id + 1]
        if start == end:
            continue

        bundle_ids = indices[start:end]
        user_vec = user_embeddings[user_id]
        bundle_vecs = bundle_embeddings[bundle_ids]
        scores = np.sum(bundle_vecs * user_vec[None, :], axis=1)
        mean_score = scores.mean(dtype=np.float32)
        adjusted = 1.0 + gamma * (scores - mean_score)
        weighted_data[start:end] = np.maximum(adjusted, eps).astype(np.float32, copy=False)

    return sp.csr_matrix((weighted_data, indices.copy(), indptr.copy()), shape=u_b_graph.shape)


class BundleTrainDataset(Dataset):
    def __init__(self, conf, u_b_pairs, u_b_graph, num_bundles, u_b_for_neg_sample, b_b_for_neg_sample, neg_sample=1):
        self.conf = conf
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.num_bundles = num_bundles
        self.neg_sample = neg_sample

        self.u_b_for_neg_sample = u_b_for_neg_sample
        self.b_b_for_neg_sample = b_b_for_neg_sample


    def __getitem__(self, index):
        conf = self.conf
        user_b, pos_bundle = self.u_b_pairs[index]
        all_bundles = [pos_bundle]

        while True:
            i = np.random.randint(self.num_bundles)
            if self.u_b_graph[user_b, i] == 0 and not i in all_bundles:                                                          
                all_bundles.append(i)                                                                                                   
                if len(all_bundles) == self.neg_sample+1:                                                                               
                    break                                                                                                               

        return torch.LongTensor([user_b]), torch.LongTensor(all_bundles)


    def __len__(self):
        return len(self.u_b_pairs)


class BundleTestDataset(Dataset):
    def __init__(self, u_b_pairs, u_b_graph, u_b_graph_train, num_users, num_bundles):
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.train_mask_u_b = u_b_graph_train

        self.num_users = num_users
        self.num_bundles = num_bundles

        self.users = torch.arange(num_users, dtype=torch.long).unsqueeze(dim=1)
        self.bundles = torch.arange(num_bundles, dtype=torch.long)


    def __getitem__(self, index):
        u_b_grd = torch.from_numpy(self.u_b_graph[index].toarray()).squeeze()
        u_b_mask = torch.from_numpy(self.train_mask_u_b[index].toarray()).squeeze()

        return index, u_b_grd, u_b_mask


    def __len__(self):
        return self.u_b_graph.shape[0]


class Datasets():
    def __init__(self, conf):
        self.conf = conf
        self.path = conf['data_path']
        self.name = conf['dataset']
        batch_size_train = conf['batch_size_train']
        batch_size_test = conf['batch_size_test']

        self.num_users, self.num_bundles, self.num_items = self.get_data_size()

        b_i_pairs, b_i_graph = self.get_bi()
        u_i_pairs, u_i_graph = self.get_ui()

        u_b_pairs_train, u_b_graph_train = self.get_ub("train")
        u_b_pairs_val, u_b_graph_val = self.get_ub("tune")
        u_b_pairs_test, u_b_graph_test = self.get_ub("test")

        u_b_for_neg_sample, b_b_for_neg_sample = None, None

        self.bundle_train_data = BundleTrainDataset(conf, u_b_pairs_train, u_b_graph_train, self.num_bundles, u_b_for_neg_sample, b_b_for_neg_sample, conf["neg_num"])
        self.bundle_val_data = BundleTestDataset(u_b_pairs_val, u_b_graph_val, u_b_graph_train, self.num_users, self.num_bundles)
        self.bundle_test_data = BundleTestDataset(u_b_pairs_test, u_b_graph_test, u_b_graph_train, self.num_users, self.num_bundles)

        self.graphs = [u_b_graph_train, u_i_graph, b_i_graph]
        self.user_observed_bundles = self.build_user_observed_bundles(u_b_graph_train)
        self.user_observed_bundle_sets = [set(bundles) for bundles in self.user_observed_bundles]
        self.weight_matrix_ub_graph = None
        self.weight_matrix_pruned_ub_graph = None
        self.weight_matrix_bottomk_ub_graph = None
        self.weight_matrix_user_edges = None
        self.weight_matrix_topk_candidates = None
        self.weight_matrix_bottomk_candidates = None
        if (
            conf.get("use_weight_matrix_rebuild", False)
            or conf.get("use_weight_matrix_prune_bottomk", False)
            or conf.get("use_weight_matrix_keep_bottomk_rebuild", False)
        ):
            self.weight_matrix_user_edges = self.get_weight_matrix_user_edges()
        if conf.get("use_weight_matrix_rebuild", False):
            self.weight_matrix_topk_candidates = self.get_weight_matrix_topk_candidates()
            if not conf.get("use_weight_matrix_random_subset", False):
                self.weight_matrix_ub_graph = self.build_weight_matrix_ub_graph()
        if conf.get("use_weight_matrix_prune_bottomk", False):
            self.weight_matrix_pruned_ub_graph = self.build_weight_matrix_pruned_ub_graph()
        if conf.get("use_weight_matrix_keep_bottomk_rebuild", False):
            self.weight_matrix_bottomk_candidates = self.get_weight_matrix_bottomk_candidates()
            if not conf.get("use_weight_matrix_keep_bottomk_random_subset", False):
                self.weight_matrix_bottomk_ub_graph = self.build_weight_matrix_bottomk_ub_graph()

        self.train_loader = DataLoader(self.bundle_train_data, batch_size=batch_size_train, shuffle=True, num_workers=10, drop_last=True)
        self.val_loader = DataLoader(self.bundle_val_data, batch_size=batch_size_test, shuffle=False, num_workers=20)
        self.test_loader = DataLoader(self.bundle_test_data, batch_size=batch_size_test, shuffle=False, num_workers=20)


    def get_data_size(self):
        name = self.name
        if "_" in name:
            name = name.split("_")[0]
        with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(name)), 'r') as f:
            return [int(s) for s in f.readline().split('\t')][:3]


    def get_bi(self):
        with open(os.path.join(self.path, self.name, 'bundle_item.txt'), 'r') as f:
            b_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(b_i_pairs, dtype=np.int32)
        values = np.ones(len(b_i_pairs), dtype=np.float32)
        b_i_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_bundles, self.num_items)).tocsr()

        print_statistics(b_i_graph, 'B-I statistics', self.conf["log_path"])

        return b_i_pairs, b_i_graph


    def get_ui(self):
        with open(os.path.join(self.path, self.name, 'user_item.txt'), 'r') as f:
            u_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(u_i_pairs, dtype=np.int32)
        values = np.ones(len(u_i_pairs), dtype=np.float32)
        u_i_graph = sp.coo_matrix( 
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_items)).tocsr()

        print_statistics(u_i_graph, 'U-I statistics', self.conf["log_path"])

        return u_i_pairs, u_i_graph


    def get_ub(self, task):
        with open(os.path.join(self.path, self.name, 'user_bundle_{}.txt'.format(task)), 'r') as f:
            u_b_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(u_b_pairs, dtype=np.int32)
        values = np.ones(len(u_b_pairs), dtype=np.float32)
        u_b_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_bundles)).tocsr()

        print_statistics(u_b_graph, f"U-B statistics in {task}", self.conf["log_path"])

        return u_b_pairs, u_b_graph


    def build_user_observed_bundles(self, u_b_graph):
        user_observed_bundles = []
        indptr = u_b_graph.indptr
        indices = u_b_graph.indices
        for user_id in range(self.num_users):
            start = indptr[user_id]
            end = indptr[user_id + 1]
            user_observed_bundles.append(indices[start:end].astype(np.int64).tolist())
        return user_observed_bundles


    def get_weight_matrix_user_edges(self):
        weight_matrix_path = self.conf.get("weight_matrix_path")
        if not weight_matrix_path:
            raise ValueError("weight_matrix_path must be provided when weight matrix logic is enabled")

        user_edges = [[] for _ in range(self.num_users)]
        with open(weight_matrix_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                user_id, bundle_id, weight = line.split()
                user_id = int(user_id)
                bundle_id = int(bundle_id)
                weight = float(weight)
                if user_id < 0 or user_id >= self.num_users:
                    raise ValueError(f"user_id {user_id} at line {line_idx} is out of range")
                if bundle_id < 0 or bundle_id >= self.num_bundles:
                    raise ValueError(f"bundle_id {bundle_id} at line {line_idx} is out of range")
                user_edges[user_id].append((weight, bundle_id))

        for user_id, edges in enumerate(user_edges):
            if not edges:
                continue
            # Keep ties deterministic by relying on Python's stable sort.
            edges.sort(key=lambda x: x[0], reverse=True)

        if not any(user_edges):
            raise ValueError("weight matrix file produced an empty candidate pool")

        return user_edges


    def get_weight_matrix_topk_candidates(self):
        rebuild_k = self.conf["rebuild_k"]
        if rebuild_k <= 0:
            raise ValueError("rebuild_k must be positive when use_weight_matrix_rebuild is enabled")
        if self.weight_matrix_user_edges is None:
            raise ValueError("weight matrix user edges are not initialized")

        user_edges = []
        for edges in self.weight_matrix_user_edges:
            user_edges.append(edges[:rebuild_k])

        if not any(user_edges):
            raise ValueError("weight matrix rebuild produced an empty candidate pool")

        return user_edges


    def get_weight_matrix_bottomk_candidates(self):
        keep_k = self.conf.get("weight_matrix_keep_bottomk_k", 0)
        if keep_k <= 0:
            raise ValueError("weight_matrix_keep_bottomk_k must be positive when use_weight_matrix_keep_bottomk_rebuild is enabled")
        if self.weight_matrix_user_edges is None:
            raise ValueError("weight matrix user edges are not initialized")

        user_edges = []
        for edges in self.weight_matrix_user_edges:
            if keep_k >= len(edges):
                user_edges.append(list(edges))
            else:
                user_edges.append(list(edges[-keep_k:]))

        if not any(user_edges):
            raise ValueError("weight matrix keep-bottom-k rebuild produced an empty candidate pool")

        return user_edges


    def build_weight_matrix_ub_graph(self, log_stats=True):
        if self.weight_matrix_topk_candidates is None:
            raise ValueError("weight matrix top-k candidates are not initialized")

        random_subset_enabled = self.conf.get("use_weight_matrix_random_subset", False)
        subset_min_k = self.conf.get("weight_matrix_random_subset_min_k", 2)
        subset_max_k = self.conf.get("weight_matrix_random_subset_max_k", 3)
        if subset_min_k <= 0 or subset_max_k <= 0:
            raise ValueError("weight_matrix_random_subset_min_k and weight_matrix_random_subset_max_k must be positive")
        if subset_min_k > subset_max_k:
            raise ValueError("weight_matrix_random_subset_min_k cannot be greater than weight_matrix_random_subset_max_k")

        u_list = []
        b_list = []
        edge_list = []
        for user_id, edges in enumerate(self.weight_matrix_topk_candidates):
            if not edges:
                continue

            selected_edges = edges
            if random_subset_enabled:
                max_selectable = min(subset_max_k, len(edges))
                min_selectable = min(subset_min_k, len(edges))
                if min_selectable > max_selectable:
                    min_selectable = max_selectable
                sample_size = np.random.randint(min_selectable, max_selectable + 1)
                sampled_indices = np.random.choice(len(edges), size=sample_size, replace=False)
                selected_edges = [edges[idx] for idx in sampled_indices]

            for _, bundle_id in selected_edges:
                u_list.append(user_id)
                b_list.append(bundle_id)
                edge_list.append(1.0)

        if not edge_list:
            raise ValueError("weight matrix rebuild produced an empty UB graph")

        weight_matrix_ub_graph = sp.coo_matrix(
            (np.array(edge_list, dtype=np.float32), (np.array(u_list, dtype=np.int32), np.array(b_list, dtype=np.int32))),
            shape=(self.num_users, self.num_bundles),
        ).tocsr()

        if log_stats:
            stats_label = 'U-B statistics from weight matrix rebuild'
            if random_subset_enabled:
                stats_label += ' (random subset)'
            print_statistics(weight_matrix_ub_graph, stats_label, self.conf["log_path"])
        return weight_matrix_ub_graph


    def build_weight_matrix_pruned_ub_graph(self, log_stats=True):
        if self.weight_matrix_user_edges is None:
            raise ValueError("weight matrix user edges are not initialized")

        prune_k = self.conf.get("weight_matrix_prune_k", 0)
        if prune_k < 0:
            raise ValueError("weight_matrix_prune_k must be non-negative")

        u_list = []
        b_list = []
        edge_list = []
        for user_id, edges in enumerate(self.weight_matrix_user_edges):
            if not edges:
                continue

            keep_num = len(edges) - prune_k
            if keep_num <= 0:
                keep_num = 1
            selected_edges = edges[:keep_num]

            for _, bundle_id in selected_edges:
                u_list.append(user_id)
                b_list.append(bundle_id)
                edge_list.append(1.0)

        if not edge_list:
            raise ValueError("weight matrix bottom-k pruning produced an empty UB graph")

        pruned_ub_graph = sp.coo_matrix(
            (np.array(edge_list, dtype=np.float32), (np.array(u_list, dtype=np.int32), np.array(b_list, dtype=np.int32))),
            shape=(self.num_users, self.num_bundles),
        ).tocsr()

        if log_stats:
            print_statistics(pruned_ub_graph, 'U-B statistics from weight matrix bottom-k pruning', self.conf["log_path"])
        return pruned_ub_graph


    def build_weight_matrix_bottomk_ub_graph(self, log_stats=True):
        if self.weight_matrix_bottomk_candidates is None:
            raise ValueError("weight matrix bottom-k candidates are not initialized")

        random_subset_enabled = self.conf.get("use_weight_matrix_keep_bottomk_random_subset", False)
        subset_min_k = self.conf.get("weight_matrix_keep_bottomk_random_subset_min_k", 2)
        subset_max_k = self.conf.get("weight_matrix_keep_bottomk_random_subset_max_k", 3)
        if subset_min_k <= 0 or subset_max_k <= 0:
            raise ValueError("weight_matrix_keep_bottomk_random_subset_min_k and weight_matrix_keep_bottomk_random_subset_max_k must be positive")
        if subset_min_k > subset_max_k:
            raise ValueError("weight_matrix_keep_bottomk_random_subset_min_k cannot be greater than weight_matrix_keep_bottomk_random_subset_max_k")

        u_list = []
        b_list = []
        edge_list = []
        for user_id, edges in enumerate(self.weight_matrix_bottomk_candidates):
            if not edges:
                continue

            selected_edges = edges
            if random_subset_enabled:
                max_selectable = min(subset_max_k, len(edges))
                min_selectable = min(subset_min_k, len(edges))
                if min_selectable > max_selectable:
                    min_selectable = max_selectable
                sample_size = np.random.randint(min_selectable, max_selectable + 1)
                sampled_indices = np.random.choice(len(edges), size=sample_size, replace=False)
                selected_edges = [edges[idx] for idx in sampled_indices]

            for _, bundle_id in selected_edges:
                u_list.append(user_id)
                b_list.append(bundle_id)
                edge_list.append(1.0)

        if not edge_list:
            raise ValueError("weight matrix keep-bottom-k rebuild produced an empty UB graph")

        bottomk_ub_graph = sp.coo_matrix(
            (np.array(edge_list, dtype=np.float32), (np.array(u_list, dtype=np.int32), np.array(b_list, dtype=np.int32))),
            shape=(self.num_users, self.num_bundles),
        ).tocsr()

        if log_stats:
            stats_label = 'U-B statistics from weight matrix keep-bottom-k rebuild'
            if random_subset_enabled:
                stats_label += ' (random subset)'
            print_statistics(bottomk_ub_graph, stats_label, self.conf["log_path"])
        return bottomk_ub_graph


class DiffusionDataset(Dataset):
    def __init__(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix

    def __getitem__(self, user_index):
        batch_user_bundle = self.adjacency_matrix[user_index]
        return batch_user_bundle, user_index

    def __len__(self):
        return self.adjacency_matrix.shape[0]
