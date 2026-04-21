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
    load_external_embedding_tensor,
    print_statistics,
    write_log,
)
from models.DCBR import DCBR
from models.LatentDiffusionRebuilder import LatentDiffusionRebuilder


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


def build_observed_bundle_batch(dataset, user_indices, device):
    batch_users = np.asarray(user_indices, dtype=np.int64)
    batch_observed = [dataset.user_observed_bundles[int(user_id)] for user_id in batch_users]
    max_len = max(len(observed) for observed in batch_observed)

    observed_indices = torch.full(
        (len(batch_users), max_len),
        -1,
        dtype=torch.long,
        device=device,
    )
    observed_mask = torch.zeros(
        (len(batch_users), max_len),
        dtype=torch.float32,
        device=device,
    )

    for row_idx, observed in enumerate(batch_observed):
        observed_tensor = torch.tensor(observed, dtype=torch.long, device=device)
        observed_indices[row_idx, :observed_tensor.shape[0]] = observed_tensor
        observed_mask[row_idx, :observed_tensor.shape[0]] = 1.0

    return {
        "user_indices": torch.tensor(batch_users, dtype=torch.long, device=device),
        "observed_indices": observed_indices,
        "observed_mask": observed_mask,
    }


def train_latent_diffusion_epoch(latent_diffusion_model, latent_diffusion_optimizer, dataset, conf, device):
    user_order = np.random.permutation(dataset.num_users)
    batch_size = conf["latent_diffusion_batch_size"]
    step_num = (dataset.num_users + batch_size - 1) // batch_size
    pbar = tqdm(range(step_num), total=step_num)
    total_loss = 0.0
    effective_steps = 0

    latent_diffusion_model.train(True)
    for step_idx in pbar:
        start = step_idx * batch_size
        end = min((step_idx + 1) * batch_size, dataset.num_users)
        batch = build_observed_bundle_batch(dataset, user_order[start:end], device)

        latent_diffusion_optimizer.zero_grad()
        loss = latent_diffusion_model.training_loss(batch)
        loss.backward()
        latent_diffusion_optimizer.step()

        total_loss += loss.detach().item()
        effective_steps += 1
        pbar.set_description(
            f'LatentDiffusion Step: {step_idx + 1}/{step_num} | loss: {loss.detach():8.4f}'
        )

    return total_loss / max(effective_steps, 1)


def rebuild_ub_graph_with_latent_diffusion(latent_diffusion_model, dataset, conf, device):
    batch_size = conf["latent_diffusion_infer_batch_size"]
    rebuild_k = conf["rebuild_k"]
    user_indices = np.arange(dataset.num_users)
    u_list = []
    b_list = []
    edge_list = []

    latent_diffusion_model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, dataset.num_users, batch_size), desc="LatentDiffusion Rebuild"):
            end = min(start + batch_size, dataset.num_users)
            batch = build_observed_bundle_batch(dataset, user_indices[start:end], device)
            selected_bundles, selected_valid = latent_diffusion_model.rebuild_topk(
                batch["user_indices"],
                batch["observed_indices"],
                rebuild_k=rebuild_k,
            )

            selected_bundles = selected_bundles.cpu().numpy()
            selected_valid = selected_valid.cpu().numpy()
            batch_users = batch["user_indices"].cpu().numpy()
            for row_idx, user_id in enumerate(batch_users):
                for col_idx in range(selected_bundles.shape[1]):
                    if not selected_valid[row_idx, col_idx]:
                        continue
                    u_list.append(int(user_id))
                    b_list.append(int(selected_bundles[row_idx, col_idx]))
                    edge_list.append(1.0)

    rebuilt_ub_graph = sp.coo_matrix(
        (
            np.array(edge_list, dtype=np.float32),
            (np.array(u_list, dtype=np.int32), np.array(b_list, dtype=np.int32)),
        ),
        shape=(dataset.num_users, dataset.num_bundles),
    ).tocsr()
    return rebuilt_ub_graph


def create_latent_diffusion_rebuilder(conf, user_embedding_tensor, bundle_embedding_tensor, device):
    latent_diffusion_model = LatentDiffusionRebuilder(
        conf,
        user_embedding_tensor.to(device),
        bundle_embedding_tensor.to(device),
    ).to(device)
    latent_diffusion_optimizer = torch.optim.Adam(
        latent_diffusion_model.parameters(),
        lr=conf["latent_diffusion_lr"],
        weight_decay=conf["latent_diffusion_weight_decay"],
    )
    return latent_diffusion_model, latent_diffusion_optimizer


def reset_latent_diffusion_rebuilder(conf, latent_diffusion_model, latent_diffusion_optimizer):
    latent_diffusion_model.reset_parameters()
    if conf.get("latent_diffusion_reset_optimizer", True):
        latent_diffusion_optimizer = torch.optim.Adam(
            latent_diffusion_model.parameters(),
            lr=conf["latent_diffusion_lr"],
            weight_decay=conf["latent_diffusion_weight_decay"],
        )
    return latent_diffusion_model, latent_diffusion_optimizer


def main():
    paras = get_cmd().__dict__
    set_seed(paras["seed"])
    conf_overall = yaml.safe_load(open("configs/overall.yaml"))
    conf_model = yaml.safe_load(open(f"configs/models/{paras['model']}.yaml"))
    print("load config file done!")

    dataset_name = paras["dataset"]
    assert paras["model"] in ["DCBR"], "Pls select models from: DCBR"
    if dataset_name != "iFashion":
        raise ValueError("The current codebase only supports the iFashion dataset.")

    conf_model = conf_model["iFashion"]
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
    use_latent_diffusion_rebuild = conf.get("use_latent_diffusion_rebuild", False)

    latent_diffusion_model = None
    latent_diffusion_optimizer = None
    if not use_latent_diffusion_rebuild:
        raise ValueError("The current codebase only supports the latent diffusion rebuild path.")

    latent_diffusion_user_embedding_tensor = load_external_embedding_tensor(
        conf["latent_diffusion_user_embedding_path"],
        dataset.num_users,
        "user",
    )
    latent_diffusion_bundle_embedding_tensor = load_external_embedding_tensor(
        conf["latent_diffusion_bundle_embedding_path"],
        dataset.num_bundles,
        "bundle",
    )
    if latent_diffusion_user_embedding_tensor.shape[1] != latent_diffusion_bundle_embedding_tensor.shape[1]:
        raise ValueError("Latent diffusion user and bundle embeddings must share the same dimension")
    latent_diffusion_model, latent_diffusion_optimizer = create_latent_diffusion_rebuilder(
        conf,
        latent_diffusion_user_embedding_tensor,
        latent_diffusion_bundle_embedding_tensor,
        device,
    )
    write_log(
        "Use latent diffusion rebuild: train a shared latent denoiser on observed bundle-set embeddings and rebuild UB graph from observed top-k anchor matches.",
        log_path,
    )

    batch_cnt = len(dataset.train_loader)
    test_interval_bs = int(batch_cnt * conf["test_interval"])

    best_metrics = init_best_metrics(conf)
    best_epoch = 0
    best_content = None
    for epoch in range(conf['epochs']):
        if (
            use_latent_diffusion_rebuild
            and conf.get("latent_diffusion_reset_after_epoch", -1) >= 0
            and epoch == conf["latent_diffusion_reset_after_epoch"] + 1
        ):
            latent_diffusion_model, latent_diffusion_optimizer = reset_latent_diffusion_rebuilder(
                conf,
                latent_diffusion_model,
                latent_diffusion_optimizer,
            )
            write_log(
                (
                    "Latent diffusion rebuild reset triggered after epoch "
                    f"{conf['latent_diffusion_reset_after_epoch']}: reinitialized the latent top-k selector"
                    + (" and optimizer." if conf.get("latent_diffusion_reset_optimizer", True) else ".")
                ),
                log_path,
            )
        latent_diffusion_loss = train_latent_diffusion_epoch(
            latent_diffusion_model,
            latent_diffusion_optimizer,
            dataset,
            conf,
            device,
        )
        write_log(f"Latent diffusion average loss at epoch {epoch}: {latent_diffusion_loss:.6f}", log_path)
        rebuilt_ub_graph = rebuild_ub_graph_with_latent_diffusion(latent_diffusion_model, dataset, conf, device)
        if epoch == 0:
            print_statistics(rebuilt_ub_graph, "U-B statistics from Latent diffusion rebuild", log_path)
        UB_propagation_graph = build_ub_propagation_graph(rebuilt_ub_graph, conf, device)

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
