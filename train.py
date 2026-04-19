import os
import yaml
import random
import argparse
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import torch
from utility import (
    Datasets,
    DiffusionDataset,
    build_weighted_cbdm_input_graph,
    load_external_embedding_tensor,
    print_statistics,
    write_log,
)
from models.DCBR import DCBR, DNN, GaussianDiffusion
from models.AnchorRebuilder import AnchorRebuilder
from models.MBMRebuilder import MBMRebuilder


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=str, default="0")
    parser.add_argument("-d", "--dataset", type=str, default="iFashion")
    parser.add_argument("-m", "--model", type=str, default="DCBR")
    parser.add_argument("-i", "--info", type=str, default="")
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()
    return args


def build_ub_propagation_graph(ub_graph, conf, device):
    adjacency_matrix = sp.bmat([
        [sp.csr_matrix((conf["num_users"], conf["num_users"])), ub_graph],
        [ub_graph.T, sp.csr_matrix((conf["num_bundles"], conf["num_bundles"]))],
    ])
    adjacency_matrix = adjacency_matrix + sp.eye(adjacency_matrix.shape[0])
    row_sum = np.array(adjacency_matrix.sum(axis=1))
    d_inv = np.power(row_sum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    degree_matrix = sp.diags(d_inv)
    norm_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocoo()
    values = norm_adjacency.data
    indices = np.vstack((norm_adjacency.row, norm_adjacency.col))
    return torch.sparse_coo_tensor(
        torch.LongTensor(indices),
        torch.FloatTensor(values),
        torch.Size(norm_adjacency.shape),
    ).to(device)


def write_analysis_lines(file_path, lines):
    with open(file_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def write_cbdm_epoch_top1_matrix(file_path, top1_by_epoch, num_users, num_epochs):
    header = ["user_id"] + [f"epoch_{epoch}" for epoch in range(num_epochs)]
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for user_id in range(num_users):
            row = [str(user_id)]
            for epoch in range(num_epochs):
                row.append(str(top1_by_epoch[epoch][user_id]))
            f.write("\t".join(row) + "\n")


def sample_mbm_batch(dataset, user_indices, conf, device):
    context_lists = []
    pos_bundles = []
    neg_bundles = []
    selected_users = []

    neg_num = conf["mbm_neg_num"]
    mask_ratio = conf["mbm_mask_ratio"]

    for user_id in user_indices:
        observed = dataset.user_observed_bundles[user_id]
        if len(observed) <= 1:
            continue

        observed = np.asarray(observed, dtype=np.int64)
        pos_pos = np.random.randint(observed.shape[0])
        pos_bundle = int(observed[pos_pos])
        remaining = np.delete(observed, pos_pos)
        if remaining.shape[0] == 0:
            continue

        extra_mask = int(np.floor(remaining.shape[0] * mask_ratio))
        extra_mask = min(extra_mask, max(0, remaining.shape[0] - 1))
        if extra_mask > 0:
            drop_indices = np.random.choice(remaining.shape[0], size=extra_mask, replace=False)
            context = np.delete(remaining, drop_indices)
        else:
            context = remaining
        if context.shape[0] == 0:
            context = remaining[:1]

        negatives = []
        observed_set = dataset.user_observed_bundle_sets[user_id]
        while len(negatives) < neg_num:
            candidate = np.random.randint(dataset.num_bundles)
            if candidate in observed_set or candidate in negatives:
                continue
            negatives.append(candidate)

        context_lists.append(context.tolist())
        pos_bundles.append(pos_bundle)
        neg_bundles.append(negatives)
        selected_users.append(user_id)

    if not selected_users:
        return None

    max_context_len = max(len(context) for context in context_lists)
    context_indices = torch.full(
        (len(selected_users), max_context_len),
        -1,
        dtype=torch.long,
        device=device,
    )
    context_mask = torch.zeros(
        (len(selected_users), max_context_len),
        dtype=torch.float32,
        device=device,
    )

    for row_idx, context in enumerate(context_lists):
        context_tensor = torch.tensor(context, dtype=torch.long, device=device)
        context_indices[row_idx, :context_tensor.shape[0]] = context_tensor
        context_mask[row_idx, :context_tensor.shape[0]] = 1.0

    return {
        "user_indices": torch.tensor(selected_users, dtype=torch.long, device=device),
        "context_indices": context_indices,
        "context_mask": context_mask,
        "pos_indices": torch.tensor(pos_bundles, dtype=torch.long, device=device),
        "neg_indices": torch.tensor(neg_bundles, dtype=torch.long, device=device),
    }


def train_mbm_epoch(mbm_model, mbm_optimizer, dataset, conf, device):
    user_order = np.random.permutation(dataset.num_users)
    batch_size = conf["mbm_batch_size"]
    step_num = (dataset.num_users + batch_size - 1) // batch_size
    pbar = tqdm(range(step_num), total=step_num)
    total_loss = 0.0
    effective_steps = 0

    mbm_model.train(True)
    for step_idx in pbar:
        start = step_idx * batch_size
        end = min((step_idx + 1) * batch_size, dataset.num_users)
        batch = sample_mbm_batch(dataset, user_order[start:end], conf, device)
        if batch is None:
            continue

        mbm_optimizer.zero_grad()
        loss = mbm_model.training_loss(batch)
        loss.backward()
        mbm_optimizer.step()

        total_loss += loss.detach().item()
        effective_steps += 1
        pbar.set_description(f'MBM Step: {step_idx + 1}/{step_num} | loss: {loss.detach():8.4f}')

    return total_loss / max(effective_steps, 1)


def rebuild_ub_graph_with_mbm(mbm_model, dataset, conf, device):
    batch_size = conf["mbm_infer_batch_size"]
    rebuild_k = conf["rebuild_k"]
    user_indices = np.arange(dataset.num_users)
    u_list = []
    b_list = []
    edge_list = []

    mbm_model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, dataset.num_users, batch_size), desc="MBM Rebuild"):
            end = min(start + batch_size, dataset.num_users)
            batch_users = user_indices[start:end]
            batch_observed = dataset.user_observed_bundles[start:end]
            max_len = max(len(observed) for observed in batch_observed)

            observed_tensor = torch.full(
                (len(batch_users), max_len),
                -1,
                dtype=torch.long,
                device=device,
            )
            for row_idx, observed in enumerate(batch_observed):
                observed_idx = torch.tensor(observed, dtype=torch.long, device=device)
                observed_tensor[row_idx, :observed_idx.shape[0]] = observed_idx

            selected_bundles, selected_valid = mbm_model.rebuild_topk(
                torch.tensor(batch_users, dtype=torch.long, device=device),
                observed_tensor,
                rebuild_k=rebuild_k,
            )

            selected_bundles = selected_bundles.cpu().numpy()
            selected_valid = selected_valid.cpu().numpy()
            for row_idx, user_id in enumerate(batch_users):
                for col_idx in range(selected_bundles.shape[1]):
                    if not selected_valid[row_idx, col_idx]:
                        continue
                    u_list.append(int(user_id))
                    b_list.append(int(selected_bundles[row_idx, col_idx]))
                    edge_list.append(1.0)

    rebuilt_ub_graph = sp.coo_matrix(
        (np.array(edge_list, dtype=np.float32), (np.array(u_list, dtype=np.int32), np.array(b_list, dtype=np.int32))),
        shape=(dataset.num_users, dataset.num_bundles),
    ).tocsr()
    return rebuilt_ub_graph


def sample_anchor_batch(dataset, user_indices, conf, device):
    anchor_users = []
    anchor_bundles = []
    pos_bundles = []
    neg_bundles = []

    neg_num = conf["anchor_neg_num"]
    for user_id in user_indices:
        observed = dataset.user_observed_bundles[user_id]
        if len(observed) <= 1:
            continue

        observed = np.asarray(observed, dtype=np.int64)
        anchor_pos = np.random.randint(observed.shape[0])
        anchor_bundle = int(observed[anchor_pos])
        remaining = np.delete(observed, anchor_pos)
        if remaining.shape[0] == 0:
            continue

        pos_bundle = int(remaining[np.random.randint(remaining.shape[0])])
        negatives = []
        observed_set = dataset.user_observed_bundle_sets[user_id]
        while len(negatives) < neg_num:
            candidate = np.random.randint(dataset.num_bundles)
            if candidate in observed_set or candidate in negatives:
                continue
            negatives.append(candidate)

        anchor_users.append(user_id)
        anchor_bundles.append(anchor_bundle)
        pos_bundles.append(pos_bundle)
        neg_bundles.append(negatives)

    if not anchor_users:
        return None

    return {
        "user_indices": torch.tensor(anchor_users, dtype=torch.long, device=device),
        "anchor_indices": torch.tensor(anchor_bundles, dtype=torch.long, device=device),
        "pos_indices": torch.tensor(pos_bundles, dtype=torch.long, device=device),
        "neg_indices": torch.tensor(neg_bundles, dtype=torch.long, device=device),
    }


def train_anchor_epoch(anchor_model, anchor_optimizer, dataset, conf, device):
    user_order = np.random.permutation(dataset.num_users)
    batch_size = conf["anchor_batch_size"]
    step_num = (dataset.num_users + batch_size - 1) // batch_size
    pbar = tqdm(range(step_num), total=step_num)
    total_loss = 0.0
    effective_steps = 0

    anchor_model.train(True)
    for step_idx in pbar:
        start = step_idx * batch_size
        end = min((step_idx + 1) * batch_size, dataset.num_users)
        batch = sample_anchor_batch(dataset, user_order[start:end], conf, device)
        if batch is None:
            continue

        anchor_optimizer.zero_grad()
        loss = anchor_model.training_loss(batch)
        loss.backward()
        anchor_optimizer.step()

        total_loss += loss.detach().item()
        effective_steps += 1
        pbar.set_description(f'Anchor Step: {step_idx + 1}/{step_num} | loss: {loss.detach():8.4f}')

    return total_loss / max(effective_steps, 1)


def rebuild_ub_graph_with_anchor(anchor_model, dataset, conf, device):
    batch_size = conf["anchor_infer_batch_size"]
    rebuild_k = conf["rebuild_k"]
    user_indices = np.arange(dataset.num_users)
    u_list = []
    b_list = []
    edge_list = []

    anchor_model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, dataset.num_users, batch_size), desc="Anchor Rebuild"):
            end = min(start + batch_size, dataset.num_users)
            batch_users = user_indices[start:end]
            batch_observed = dataset.user_observed_bundles[start:end]
            max_len = max(len(observed) for observed in batch_observed)

            observed_tensor = torch.full(
                (len(batch_users), max_len),
                -1,
                dtype=torch.long,
                device=device,
            )
            for row_idx, observed in enumerate(batch_observed):
                observed_idx = torch.tensor(observed, dtype=torch.long, device=device)
                observed_tensor[row_idx, :observed_idx.shape[0]] = observed_idx

            selected_bundles, selected_valid = anchor_model.rebuild_topk(
                torch.tensor(batch_users, dtype=torch.long, device=device),
                observed_tensor,
                rebuild_k=rebuild_k,
            )

            selected_bundles = selected_bundles.cpu().numpy()
            selected_valid = selected_valid.cpu().numpy()
            for row_idx, user_id in enumerate(batch_users):
                for col_idx in range(selected_bundles.shape[1]):
                    if not selected_valid[row_idx, col_idx]:
                        continue
                    u_list.append(int(user_id))
                    b_list.append(int(selected_bundles[row_idx, col_idx]))
                    edge_list.append(1.0)

    rebuilt_ub_graph = sp.coo_matrix(
        (np.array(edge_list, dtype=np.float32), (np.array(u_list, dtype=np.int32), np.array(b_list, dtype=np.int32))),
        shape=(dataset.num_users, dataset.num_bundles),
    ).tocsr()
    return rebuilt_ub_graph


def main():
    paras = get_cmd().__dict__
    set_seed(paras["seed"])
    conf_overall = yaml.safe_load(open("configs/overall.yaml"))
    conf_model = yaml.safe_load(open(f"configs/models/{paras['model']}.yaml"))
    print("load config file done!")

    dataset_name = paras["dataset"]
    assert paras["model"] in ["DCBR"], "Pls select models from: DCBR"

    if "_" in dataset_name:
        conf_model = conf_model[dataset_name.split("_")[0]]
    else:
        conf_model = conf_model[dataset_name]
    conf = {**conf_model, **conf_overall, **paras}
    conf["dataset"] = dataset_name
    conf["model"] = paras["model"]

    log_path = f"./logs/{conf['dataset']}/{conf['model']}"
    checkpoint_model_path = f"./checkpoints/{conf['dataset']}/{conf['model']}"
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir(checkpoint_model_path):
        os.makedirs(checkpoint_model_path)

    setting = conf["model"] + "-" + conf["dataset"]
    if conf["info"] != "":
        setting = setting + "-" + conf["info"]
    log_path = log_path + "/" + setting + ".log"
    checkpoint_model_path = checkpoint_model_path + "/" + setting + ".pth"
    conf["log_path"] = log_path
    conf["checkpoint_model_path"] = checkpoint_model_path

    analysis_dir = f"./analysis/{conf['dataset']}/{conf['model']}"
    if not os.path.isdir(analysis_dir):
        os.makedirs(analysis_dir)
    conf["cbdm_final_ranking_path"] = f"{analysis_dir}/{setting}-cbdm-final-observed-ranking.txt"
    conf["cbdm_epoch_top1_path"] = f"{analysis_dir}/{setting}-cbdm-epoch-top1.txt"
    
    dataset = Datasets(conf)
    write_log(conf, log_path)

    conf["gpu"] = paras["gpu"]
    conf["info"] = paras["info"]
    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items
    os.environ['CUDA_VISIBLE_DEVICES'] = conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device
    
    if conf['model'] == 'DCBR':
        model = DCBR(conf, dataset.graphs).to(device)
    else:
        raise ValueError(f"Unimplemented model {conf['model']}")
    optimizer = torch.optim.Adam(model.parameters(), lr=conf["lr"], weight_decay=0)
    use_weight_matrix_rebuild = conf.get("use_weight_matrix_rebuild", False)
    use_weight_matrix_prune_bottomk = conf.get("use_weight_matrix_prune_bottomk", False)
    use_weight_matrix_keep_bottomk_rebuild = conf.get("use_weight_matrix_keep_bottomk_rebuild", False)
    use_mbm_rebuild = conf.get("use_mbm_rebuild", False)
    use_anchor_rebuild = conf.get("use_anchor_rebuild", False)

    enabled_weight_matrix_branches = sum(
        int(flag)
        for flag in (
            use_weight_matrix_rebuild,
            use_weight_matrix_prune_bottomk,
            use_weight_matrix_keep_bottomk_rebuild,
            use_mbm_rebuild,
            use_anchor_rebuild,
        )
    )
    if enabled_weight_matrix_branches > 1:
        raise ValueError(
            "Only one rebuild main branch can be enabled at a time: "
            "use_weight_matrix_rebuild, use_weight_matrix_prune_bottomk, "
            "use_weight_matrix_keep_bottomk_rebuild, use_mbm_rebuild, use_anchor_rebuild"
        )

    denoise_model = None
    diffusion_model = None
    cbdm_optimizer = None
    mbm_model = None
    mbm_optimizer = None
    anchor_model = None
    anchor_optimizer = None
    static_ub_propagation_graph = None
    use_blcc = conf.get("use_blcc", True)
    export_cbdm_analysis = conf.get("export_cbdm_analysis", False)
    use_observed_ub_rebuild_mask = conf.get("use_observed_ub_rebuild_mask", False)
    use_weight_matrix_random_subset = conf.get("use_weight_matrix_random_subset", False)
    use_weight_matrix_keep_bottomk_random_subset = conf.get("use_weight_matrix_keep_bottomk_random_subset", False)
    use_weighted_cbdm_input = conf.get("use_weighted_cbdm_input", False)
    cbdm_input_graph = dataset.graphs[0]
    if use_weighted_cbdm_input and enabled_weight_matrix_branches > 0:
        write_log(
            "weighted CBDM input is enabled in config, but the current run uses a non-CBDM rebuild branch; the weighted input will be ignored.",
            log_path,
        )
    if use_weight_matrix_prune_bottomk:
        static_ub_propagation_graph = build_ub_propagation_graph(dataset.weight_matrix_pruned_ub_graph, conf, device)
        write_log("Use weight matrix bottom-k pruning rebuild: skip CBDM training and drop the lowest-weight observed U-B edges per user.", log_path)
    elif use_weight_matrix_keep_bottomk_rebuild:
        if use_weight_matrix_keep_bottomk_random_subset:
            write_log("Use weight matrix keep-bottom-k rebuild with epoch-wise random subset sampling from the bottom-k candidates.", log_path)
        else:
            static_ub_propagation_graph = build_ub_propagation_graph(dataset.weight_matrix_bottomk_ub_graph, conf, device)
            write_log("Use weight matrix keep-bottom-k rebuild: skip CBDM training and rebuild UB graph from the bottom-k observed U-B edges.", log_path)
    elif use_weight_matrix_rebuild:
        if use_weight_matrix_random_subset:
            write_log("Use weight matrix rebuild with epoch-wise random subset sampling from weight matrix top-k candidates.", log_path)
        else:
            static_ub_propagation_graph = build_ub_propagation_graph(dataset.weight_matrix_ub_graph, conf, device)
            write_log("Use weight matrix rebuild: skip CBDM training and rebuild UB graph from weight matrix top-k.", log_path)
    elif use_mbm_rebuild:
        user_embedding_tensor = load_external_embedding_tensor(
            conf["mbm_user_embedding_path"],
            dataset.num_users,
            "user",
        )
        bundle_embedding_tensor = load_external_embedding_tensor(
            conf["mbm_bundle_embedding_path"],
            dataset.num_bundles,
            "bundle",
        )
        if user_embedding_tensor.shape[1] != bundle_embedding_tensor.shape[1]:
            raise ValueError("MBM user and bundle embeddings must share the same dimension")
        mbm_model = MBMRebuilder(conf, user_embedding_tensor.to(device), bundle_embedding_tensor.to(device)).to(device)
        mbm_optimizer = torch.optim.Adam(
            mbm_model.parameters(),
            lr=conf["mbm_lr"],
            weight_decay=conf["mbm_weight_decay"],
        )
        write_log("Use MBM rebuild: train a masked bundle selector on external LightGCN embeddings and rebuild UB graph from observed top-k anchors.", log_path)
    elif use_anchor_rebuild:
        user_embedding_tensor = load_external_embedding_tensor(
            conf["anchor_user_embedding_path"],
            dataset.num_users,
            "user",
        )
        bundle_embedding_tensor = load_external_embedding_tensor(
            conf["anchor_bundle_embedding_path"],
            dataset.num_bundles,
            "bundle",
        )
        if user_embedding_tensor.shape[1] != bundle_embedding_tensor.shape[1]:
            raise ValueError("Anchor user and bundle embeddings must share the same dimension")
        anchor_model = AnchorRebuilder(conf, user_embedding_tensor.to(device), bundle_embedding_tensor.to(device)).to(device)
        anchor_optimizer = torch.optim.Adam(
            anchor_model.parameters(),
            lr=conf["anchor_lr"],
            weight_decay=conf["anchor_weight_decay"],
        )
        write_log("Use Anchor rebuild: train an anchor-conditioned bundle reconstructor on external LightGCN embeddings and rebuild UB graph from observed top-k anchors.", log_path)
    else:
        if use_weighted_cbdm_input:
            cbdm_user_embeddings = load_external_embedding_tensor(
                conf["weighted_cbdm_user_embedding_path"],
                dataset.num_users,
                "user",
            )
            cbdm_bundle_embeddings = load_external_embedding_tensor(
                conf["weighted_cbdm_bundle_embedding_path"],
                dataset.num_bundles,
                "bundle",
            )
            cbdm_input_graph = build_weighted_cbdm_input_graph(
                dataset.graphs[0],
                cbdm_user_embeddings.cpu().numpy(),
                cbdm_bundle_embeddings.cpu().numpy(),
                conf["weighted_cbdm_input_gamma"],
                conf["weighted_cbdm_input_eps"],
            )
            write_log(
                "Use weighted CBDM input: observed U-B edges are reweighted by "
                "max(eps, 1 + gamma * (dot(u, b) - user_mean_dot)) before diffusion.",
                log_path,
            )
            write_log(
                (
                    f"Weighted CBDM input params | gamma={conf['weighted_cbdm_input_gamma']} | "
                    f"eps={conf['weighted_cbdm_input_eps']} | "
                    f"user_emb={conf['weighted_cbdm_user_embedding_path']} | "
                    f"bundle_emb={conf['weighted_cbdm_bundle_embedding_path']}"
                ),
                log_path,
            )
            write_log(
                (
                    f"Weighted CBDM input stats | min={cbdm_input_graph.data.min():.6f} | "
                    f"max={cbdm_input_graph.data.max():.6f} | "
                    f"mean={cbdm_input_graph.data.mean():.6f} | "
                    f"std={cbdm_input_graph.data.std():.6f}"
                ),
                log_path,
            )
        else:
            write_log("Use original binary CBDM input: observed U-B edges keep value 1.", log_path)
        if use_observed_ub_rebuild_mask:
            write_log("Use observed-only CBDM rebuild: top-k is selected only from observed train U-B interactions.", log_path)
        write_log(f"CBDM BLCC enabled: {use_blcc}", log_path)
        if export_cbdm_analysis:
            write_log("CBDM analysis export enabled: will dump final observed-edge ranking and per-epoch observed-edge top1.", log_path)
        # Conditional Bundle Diffusion Model (CBDM)
        out_dims = conf["dims"] + [conf["num_bundles"]]
        in_dims = out_dims[::-1]
        denoise_model = DNN(in_dims, out_dims, conf["time_emb_dim"], norm=conf["norm"]).to(device)
        diffusion_model = GaussianDiffusion(conf["noise_scale"], conf["noise_min"], conf["noise_max"], conf["steps"]).to(device)
        cbdm_optimizer = torch.optim.Adam(denoise_model.parameters(), lr=conf["lr"], weight_decay=0)
    if export_cbdm_analysis and (
        use_weight_matrix_rebuild
        or use_weight_matrix_prune_bottomk
        or use_weight_matrix_keep_bottomk_rebuild
        or use_mbm_rebuild
        or use_anchor_rebuild
    ):
        write_log("CBDM analysis export requested, but CBDM is skipped by the current non-CBDM rebuild branch. Export will be ignored.", log_path)
        export_cbdm_analysis = False

    batch_cnt = len(dataset.train_loader)
    test_interval_bs = int(batch_cnt * conf["test_interval"])

    best_metrics = init_best_metrics(conf)
    best_epoch = 0
    best_content = None
    cbdm_epoch_top1_by_epoch = {}
    cbdm_final_ranking_lines = []
    for epoch in range(conf['epochs']):
        if use_weight_matrix_prune_bottomk:
            UB_propagation_graph = static_ub_propagation_graph
        elif use_weight_matrix_keep_bottomk_rebuild:
            if use_weight_matrix_keep_bottomk_random_subset:
                sampled_weight_matrix_bottomk_ub_graph = dataset.build_weight_matrix_bottomk_ub_graph(log_stats=(epoch == 0))
                UB_propagation_graph = build_ub_propagation_graph(sampled_weight_matrix_bottomk_ub_graph, conf, device)
            else:
                UB_propagation_graph = static_ub_propagation_graph
        elif use_weight_matrix_rebuild:
            if use_weight_matrix_random_subset:
                sampled_weight_matrix_ub_graph = dataset.build_weight_matrix_ub_graph(log_stats=(epoch == 0))
                UB_propagation_graph = build_ub_propagation_graph(sampled_weight_matrix_ub_graph, conf, device)
            else:
                UB_propagation_graph = static_ub_propagation_graph
        elif use_mbm_rebuild:
            mbm_loss = train_mbm_epoch(mbm_model, mbm_optimizer, dataset, conf, device)
            write_log(f"MBM average loss at epoch {epoch}: {mbm_loss:.6f}", log_path)
            rebuilt_ub_graph = rebuild_ub_graph_with_mbm(mbm_model, dataset, conf, device)
            if epoch == 0:
                print_statistics(rebuilt_ub_graph, "U-B statistics from MBM rebuild", log_path)
            UB_propagation_graph = build_ub_propagation_graph(rebuilt_ub_graph, conf, device)
        elif use_anchor_rebuild:
            anchor_loss = train_anchor_epoch(anchor_model, anchor_optimizer, dataset, conf, device)
            write_log(f"Anchor average loss at epoch {epoch}: {anchor_loss:.6f}", log_path)
            rebuilt_ub_graph = rebuild_ub_graph_with_anchor(anchor_model, dataset, conf, device)
            if epoch == 0:
                print_statistics(rebuilt_ub_graph, "U-B statistics from Anchor rebuild", log_path)
            UB_propagation_graph = build_ub_propagation_graph(rebuilt_ub_graph, conf, device)
        else:
            ######### Denoising Uer-Bundle Graph ###########
            diffusionDataset = DiffusionDataset(torch.FloatTensor(cbdm_input_graph.toarray()))
            diffusionLoader = torch.utils.data.DataLoader(diffusionDataset, batch_size=conf["batch_size_train"], shuffle=True, num_workers=0)
            total_steps = (diffusionDataset.__len__() + conf["batch_size_train"] - 1) // conf["batch_size_train"]
            pbar_diffusion = tqdm(enumerate(diffusionLoader), total=total_steps)
            for i, batch in pbar_diffusion:
                batch_user_bundle, batch_user_index = batch
                batch_user_bundle, batch_user_index = batch_user_bundle.to(device), batch_user_index.to(device)
                uEmbeds = model.getUserEmbeds().detach() if use_blcc else None
                bEmbeds = model.getBundleEmbeds().detach() if use_blcc else None
                
                cbdm_optimizer.zero_grad()
                elbo_loss, blcc_loss = diffusion_model.training_CBDM_losses(
                    denoise_model, batch_user_bundle, uEmbeds, bEmbeds, batch_user_index, use_blcc=use_blcc
                )
                if use_blcc:
                    blcc_loss *= conf["lambda_0"]
                loss = elbo_loss + blcc_loss
                loss.backward()
                cbdm_optimizer.step()
                
                loss_scalar = loss.detach()
                elbo_loss_scalar = elbo_loss.detach()
                blcc_loss_scalar = blcc_loss.detach()
                pbar_diffusion.set_description(f'Diffusion Step: {i+1}/{total_steps} | loss: {loss_scalar:8.4f} | elbo_loss: {elbo_loss_scalar:8.4f} | blcc_loss: {blcc_loss_scalar:8.4f}')

            with torch.no_grad():
                u_list_ub = []
                b_list_ub = []
                edge_list_ub = []
                for _, batch in enumerate(diffusionLoader):
                    batch_user_bundle, batch_user_index = batch
                    batch_user_bundle, batch_user_index = batch_user_bundle.to(device), batch_user_index.to(device)
                    denoised_batch = diffusion_model.p_sample(denoise_model, batch_user_bundle, conf["sampling_steps"], conf["sampling_noise"])

                    if export_cbdm_analysis:
                        observed_mask_for_analysis = batch_user_bundle > 0
                        epoch_top1_map = cbdm_epoch_top1_by_epoch.setdefault(epoch, {})
                        for row_idx in range(batch_user_index.shape[0]):
                            observed_indices = torch.nonzero(observed_mask_for_analysis[row_idx], as_tuple=False).view(-1)
                            observed_scores = denoised_batch[row_idx, observed_indices]
                            sorted_scores, sorted_order = torch.sort(observed_scores, descending=True)
                            sorted_bundle_indices = observed_indices[sorted_order]

                            user_id = int(batch_user_index[row_idx].item())
                            top1_bundle_id = int(sorted_bundle_indices[0].item())
                            epoch_top1_map[user_id] = top1_bundle_id

                            if epoch == conf["epochs"] - 1:
                                for rank_idx in range(sorted_bundle_indices.shape[0]):
                                    bundle_id = int(sorted_bundle_indices[rank_idx].item())
                                    score = float(sorted_scores[rank_idx].item())
                                    cbdm_final_ranking_lines.append(
                                        f"{user_id}\t{bundle_id}\t{rank_idx + 1}\t{score:.10f}"
                                    )

                    if use_observed_ub_rebuild_mask:
                        observed_mask = batch_user_bundle > 0
                        denoised_batch = denoised_batch.masked_fill(~observed_mask, -1e8)
                    _, indices_ = torch.topk(denoised_batch, k=conf["rebuild_k"])
                    for i in range(batch_user_index.shape[0]):
                        for j in range(indices_[i].shape[0]): 
                            bundle_idx = int(indices_[i][j].cpu().numpy())
                            if use_observed_ub_rebuild_mask and batch_user_bundle[i, bundle_idx].item() <= 0:
                                continue
                            u_list_ub.append(int(batch_user_index[i].cpu().numpy()))
                            b_list_ub.append(bundle_idx)
                            edge_list_ub.append(1.0)
                denoised_ub_mat = sp.coo_matrix(
                    (np.array(edge_list_ub), (np.array(u_list_ub), np.array(b_list_ub))),
                    shape=(conf["num_users"], conf["num_bundles"]),
                    dtype=np.float32,
                )
                UB_propagation_graph = build_ub_propagation_graph(denoised_ub_mat, conf, device)
                
            ################################################

        epoch_anchor = epoch * batch_cnt
        pbar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader))
        for batch_i, batch in pbar:
            model.train(True)
            optimizer.zero_grad()
            batch = [x.to(device) for x in batch]
            batch_anchor = epoch_anchor + batch_i

            bpr_loss, cl_loss = model(UB_propagation_graph, batch)
            loss = bpr_loss + cl_loss
            loss.backward()
            optimizer.step()

            loss_scalar = loss.detach()
            bpr_loss_scalar = bpr_loss.detach()
            cl_loss_scalar = cl_loss.detach()
            pbar.set_description(f'epoch: {epoch:3d} | loss: {loss_scalar:8.4f} | bpr_loss: {bpr_loss_scalar:8.4f} | cl_loss: {cl_loss_scalar:8.4f}')

            if (batch_anchor + 1) % test_interval_bs == 0:
                metrics = {}
                metrics["val"] = test(model, dataset.val_loader, conf)
                metrics["test"] = test(model, dataset.test_loader, conf)
                best_metrics, best_epoch, best_content = log_metrics(conf, model, metrics, log_path, checkpoint_model_path, epoch, best_metrics, best_epoch, best_content)
    if export_cbdm_analysis:
        write_analysis_lines(conf["cbdm_final_ranking_path"], cbdm_final_ranking_lines)
        write_cbdm_epoch_top1_matrix(conf["cbdm_epoch_top1_path"], cbdm_epoch_top1_by_epoch, conf["num_users"], conf["epochs"])
        write_log(f"CBDM final observed-edge ranking exported to: {conf['cbdm_final_ranking_path']}", log_path)
        write_log(f"CBDM per-epoch observed-edge top1 exported to: {conf['cbdm_epoch_top1_path']}", log_path)
    write_log("="*26 + " BEST " + "="*26, log_path)
    write_log(best_content, log_path)


def init_best_metrics(conf):
    best_metrics = {}
    best_metrics["val"] = {}
    best_metrics["test"] = {}
    for key in best_metrics:
        best_metrics[key]["recall"] = {}
        best_metrics[key]["ndcg"] = {}
    for topk in conf['topk']:
        for key, res in best_metrics.items():
            for metric in res:
                best_metrics[key][metric][topk] = 0

    return best_metrics


def form_content(epoch, val_results, test_results, ks):
    content = f'     Epoch|'
    for k in ks:
        content += f' Recall@{k} |  NDCG@{k}  |'
    content += '\n'
    val_content = f'{epoch:10d}|'
    val_results_recall = val_results['recall']
    val_results_ndcg = val_results['ndcg']
    for k in ks:
        val_content += f'   {val_results_recall[k]:.4f}  |'
        val_content += f'   {val_results_ndcg[k]:.4f}  |'
    content += val_content + '\n'
    test_content = f'{epoch:10d}|'
    test_results_recall = test_results['recall']
    test_results_ndcg = test_results['ndcg']
    for k in ks:
        test_content += f'   {test_results_recall[k]:.4f}  |'
        test_content += f'   {test_results_ndcg[k]:.4f}  |'
    content += test_content
    return content


def log_metrics(conf, model, metrics, log_path, checkpoint_model_path, epoch, best_metrics, best_epoch, best_content):
    content = form_content(epoch, metrics["val"], metrics["test"], conf["topk"])
    write_log(content, log_path)

    topk_ = 20
    crt = f"top{topk_} as the final evaluation standard"
    write_log(crt, log_path)
    if metrics["val"]["recall"][topk_] > best_metrics["val"]["recall"][topk_] and metrics["val"]["ndcg"][topk_] > best_metrics["val"]["ndcg"][topk_]:
        state_dict = {
            "conf": conf,
            "cur_epoch": epoch,
            "content": content,
            "state_dict": model.state_dict(),
            "other_params": model.other_params()
        }
        torch.save(state_dict, checkpoint_model_path, pickle_protocol=4)
        saved = f"save the best checkpoint at the end of epoch {epoch}"
        write_log(saved, log_path)
        best_epoch = epoch
        best_content = content
        for topk in conf['topk']:
            for key, res in best_metrics.items():
                for metric in res:
                    best_metrics[key][metric][topk] = metrics[key][metric][topk]

    return best_metrics, best_epoch, best_content


def test(model, dataloader, conf):
    tmp_metrics = {}
    for m in ["recall", "ndcg"]:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]

    device = conf["device"]
    model.eval()
    rs = model.propagate(test=True)
    for users, ground_truth_u_b, train_mask_u_b in dataloader:
        pred_b = model.evaluate(rs, users.to(device))
        pred_b -= 1e8 * train_mask_u_b.to(device)
        tmp_metrics = get_metrics(tmp_metrics, ground_truth_u_b.to(device), pred_b, conf["topk"])

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]

    return metrics


def get_metrics(metrics, grd, pred, topks):
    tmp = {"recall": {}, "ndcg": {}}
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)

        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        tmp["ndcg"][topk] = get_ndcg(pred, grd, is_hit, topk)

    for m, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for i, x in enumerate(res):
                metrics[m][topk][i] += x

    return metrics


def get_recall(pred, grd, is_hit, topk):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    # remove those test cases who don't have any positive items
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt / (num_pos + epsilon)).sum().item()

    return [nomina, denorm]


def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        hit = hit / torch.log2(torch.arange(2, topk + 2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, dtype=torch.float, device=device)
        hit[:num_pos] = 1
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1 + topk, dtype=torch.float, device=device)
    IDCGs[0] = 1  # avoid 0/0
    for i in range(1, topk + 1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)

    idcg = IDCGs[num_pos]
    ndcg = dcg / idcg.to(device)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()

    return [nomina, denorm]


if __name__ == "__main__":
    main()
