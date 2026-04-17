import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import core_runtime as rt


def match_loss(gw_syn, gw_real, dis_metric):
    device = gw_real[0].device if len(gw_real) > 0 else torch.device('cpu')
    dis = torch.tensor(0.0, device=device)
    if dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)
    else:
        exit('DC error: unknown distance function')
    return dis

def distance_wb(gwr, gws):
    shape = gwr.shape

    # TODO: output node!!!!
    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis
    
class BOD(rt.GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(BOD, self).__init__(conf, training_set, test_set)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trainmodel = self.config['trainmodel']
        self.generator = self.config['generator']
        self.datasetname = self.config['dataset.name']
        self.weight_mode = self.config['weight.mode'] if self.config.contain('weight.mode') else 'original'
        self.config.config['weight.mode'] = self.weight_mode
        if self.weight_mode not in {'original', 'per_sample'}:
            raise ValueError(f"Unsupported weight.mode: {self.weight_mode}")
        args = rt.OptionConf(self.config['GM_AU'])
        self.generator_lr = float(args['-generator_lr'])
        self.generator_reg = float(args['-generator_reg'])
        self.generator_emb_size = int(args['-generator_emb_size'])
        self.weight_bpr = float(args['-weight_bpr'])
        self.weight_alignment = float(args['-weight_alignment'])
        self.weight_uniformity = float(args['-weight_uniformity'])
        self.outer_loop = int(args['-outer_loop'])
        self.inner_loop = int(args['-inner_loop'])
        self.outer_batch_size = int(args['-outer_batch_size']) if args.contain('-outer_batch_size') else min(self.batch_size, 128)
        self.lightgcn_layers = 2
        if self.config.contain('LightGCN'):
            lightgcn_args = rt.OptionConf(self.config['LightGCN'])
            if lightgcn_args.contain('-n_layer'):
                self.lightgcn_layers = int(lightgcn_args['-n_layer'])
        if self.trainmodel == "MF":
            self.model = Matrix_Factorization(self.data, self.emb_size)
        elif self.trainmodel == "NCF":
            self.model = NCFEncoder(self.data, 32, 3, [1,5,2,1])
        elif self.trainmodel == "LightGCN":
            self.model = LGCN_Encoder(self.data, self.emb_size, self.lightgcn_layers)
        elif self.trainmodel == "NGCF":
            self.model = NGCF_Encoder(self.data, self.emb_size,2)
        elif self.trainmodel == "SimGCL":
            self.model = SimGCL_Encoder(self.data, self.emb_size, 0.1, 2)
        elif self.trainmodel == 'SGL':
            self.model = SGL_Encoder(self.data, self.emb_size, 0.1, 2, 0.2, 2)
        else:
            print('Wrong trainmodel parameters!')
            exit(-1)

        self.model_generator = GraphGenerator_VAE(self.data, self.generator_emb_size)
        self.export_batch_size = 65536

    def print_model_info(self):
        super(BOD, self).print_model_info()
        print('Weight Mode:', self.weight_mode)

    def train(self):
        # Sparse graph propagation plus higher-order gradients is already memory-heavy.
        # Keeping anomaly mode enabled in normal training makes CUDA sparse backward much
        # more fragile and is only useful while debugging.
        torch.autograd.set_detect_anomaly(False)
        model = self.model.to(self.device)
        model_parameters = list(model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        
        model_generator = self.model_generator.to(self.device)
        optimizer_generator = torch.optim.Adam(model_generator.parameters(), lr=self.generator_lr)

        ol_batch_size = self.outer_batch_size


        for epoch_iter in range(self.maxEpoch):
            print("start model training...")
            for inner_iter in range(self.inner_loop):
                if self.trainmodel == "SGL":
                    dropped_adj1 = model.graph_reconstruction()
                    dropped_adj2 = model.graph_reconstruction()
                for n, batch in enumerate(rt.next_batch_pairwise(self.data, self.batch_size)):
                    u_idx, pos_i_idx, neg_i_idx = batch
                    u_idx = torch.as_tensor(u_idx, device=self.device, dtype=torch.long)
                    pos_i_idx = torch.as_tensor(pos_i_idx, device=self.device, dtype=torch.long)
                    neg_i_idx = torch.as_tensor(neg_i_idx, device=self.device, dtype=torch.long)
                    
                    model.train()
                    rec_user_emb, rec_item_emb = model()
                    pos_user_emb_syn = rec_user_emb.index_select(0, u_idx)
                    pos_item_emb_syn = rec_item_emb.index_select(0, pos_i_idx)
                    neg_item_emb_syn = rec_item_emb.index_select(0, neg_i_idx)

                    # The inner loop only updates the recommender. The generator acts as
                    # a fixed scorer here, so we avoid building an unnecessary autograd graph.
                    model_generator.eval()
                    with torch.no_grad():
                        A_weight_inner_full_detach = model_generator(pos_user_emb_syn, pos_item_emb_syn)
                        A_weight_inner_full_neg_detach = model_generator(pos_user_emb_syn, neg_item_emb_syn)
                    
                    bpr_inner = rt.bpr_loss_weight(
                        pos_user_emb_syn,
                        pos_item_emb_syn,
                        neg_item_emb_syn,
                        A_weight_inner_full_detach,
                        A_weight_inner_full_neg_detach,
                        mode=self.weight_mode,
                    )
                    alignment_inner = rt.alignment_loss_weight_1(
                        pos_user_emb_syn,
                        pos_item_emb_syn,
                        A_weight_inner_full_detach,
                        mode=self.weight_mode,
                    )

                    uniformity_inner = (
                        rt.uniformity_loss(pos_user_emb_syn) +
                        rt.uniformity_loss(pos_user_emb_syn) +
                        rt.uniformity_loss(pos_item_emb_syn) +
                        rt.uniformity_loss(neg_item_emb_syn)
                    ) / 4
                    if self.trainmodel == "SimGCL":
                        cl_loss = 0.005 * model.cal_cl_loss([u_idx,pos_i_idx])
                    elif self.trainmodel == "SGL":
                        cl_loss = 0.005 * model.cal_cl_loss([u_idx,pos_i_idx],dropped_adj1,dropped_adj2)
                    else:
                        cl_loss = 0

                    batch_loss_inner = self.weight_bpr * bpr_inner + self.weight_alignment * alignment_inner + self.weight_uniformity * uniformity_inner + cl_loss
                    optimizer.zero_grad(set_to_none=True)
                    batch_loss_inner.backward()
                    optimizer.step()
                    if n % 1000 == 0:
                        print('epoch:', epoch_iter, 'inner_training_iter:', inner_iter, 'inner_batch:', n)
                        print('inner_batch_loss:', batch_loss_inner.item())
                    del u_idx, pos_i_idx, neg_i_idx, pos_user_emb_syn, pos_item_emb_syn, neg_item_emb_syn
                    del A_weight_inner_full_detach, A_weight_inner_full_neg_detach
                    del rec_user_emb, rec_item_emb, bpr_inner, alignment_inner, uniformity_inner, batch_loss_inner
                    if self.device.type == 'cuda' and (n + 1) % 1000 == 0:
                        torch.cuda.empty_cache()
                
                with torch.no_grad():
                    self.user_emb, self.item_emb = (emb.detach() for emb in self.model())
                self.fast_evaluation(epoch_iter)
                if self.device.type == 'cuda':
                    self.user_emb = self.user_emb.cpu()
                    self.item_emb = self.item_emb.cpu()
                    torch.cuda.empty_cache()
                if self.device.type == 'cuda' and (epoch_iter + 1) % 2 == 0:
                    torch.cuda.empty_cache()
            self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

            print("start generator training...")
            for ol_iter in range(self.outer_loop):
                loss = torch.tensor(0.0, device=self.device)
                model.eval()
                model_generator.train()
                batch_ol = rt.sample_batch_pairwise(self.data, ol_batch_size)
                u_idx_ol, i_idx_ol, j_idx_ol = batch_ol
                u_idx_ol = torch.as_tensor(u_idx_ol, device=self.device, dtype=torch.long)
                i_idx_ol = torch.as_tensor(i_idx_ol, device=self.device, dtype=torch.long)
                j_idx_ol = torch.as_tensor(j_idx_ol, device=self.device, dtype=torch.long)
                pos_u_idx_ol = u_idx_ol
                pos_i_idx_ol = i_idx_ol
                neg_i_idx_ol = j_idx_ol   
                
                rec_user_emb, rec_item_emb = model()
                user_emb_ol = rec_user_emb.index_select(0, u_idx_ol)
                item_emb_ol = rec_item_emb.index_select(0, i_idx_ol)
                pos_user_emb_ol = rec_user_emb.index_select(0, pos_u_idx_ol)
                pos_item_emb_ol = rec_item_emb.index_select(0, pos_i_idx_ol)
                neg_item_emb_ol = rec_item_emb.index_select(0, neg_i_idx_ol)

                A_weight_user_item_pos = model_generator(pos_user_emb_ol, pos_item_emb_ol)
                A_weight_user_item_neg = model_generator(pos_user_emb_ol, neg_item_emb_ol)
                bpr_loss_ol = rt.bpr_loss_weight(
                    pos_user_emb_ol,
                    pos_item_emb_ol,
                    neg_item_emb_ol,
                    A_weight_user_item_pos,
                    A_weight_user_item_neg,
                    mode=self.weight_mode,
                )
                gw_real = torch.autograd.grad(bpr_loss_ol, model_parameters, retain_graph=True, create_graph=True)

                A_weight_user_item = model_generator(user_emb_ol, item_emb_ol)                
                alignment_syn_ol = rt.alignment_loss_weight_1(
                    user_emb_ol,
                    item_emb_ol,
                    A_weight_user_item,
                    mode=self.weight_mode,
                )

                gw_syn = torch.autograd.grad(alignment_syn_ol, model_parameters, create_graph=True)
                loss = match_loss(gw_real, gw_syn, 'ours')

                loss_reg = rt.l2_reg_loss(self.generator_reg, user_emb_ol, item_emb_ol)
                loss = loss + loss_reg

                optimizer_generator.zero_grad(set_to_none=True)
                loss.backward()
                optimizer_generator.step()
                print('epoch:', epoch_iter, 'outer_training_iter:', ol_iter, 'loss_real:', bpr_loss_ol.item(), 'loss_syn:', alignment_syn_ol.item())
                del u_idx_ol, i_idx_ol, j_idx_ol, user_emb_ol, item_emb_ol, pos_user_emb_ol, pos_item_emb_ol, neg_item_emb_ol
                del A_weight_user_item_pos, A_weight_user_item_neg, A_weight_user_item
                del rec_user_emb, rec_item_emb, bpr_loss_ol, alignment_syn_ol, gw_real, gw_syn, loss_reg, loss
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            model_generator.eval()
            if epoch_iter == self.maxEpoch - 1:            
                break

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        self._restore_best_generator()
        self.export_training_ub_weights()

    def save(self):
        with torch.no_grad():
            best_user_emb, best_item_emb = self.model.forward()
            self.best_user_emb = best_user_emb.detach().cpu()
            self.best_item_emb = best_item_emb.detach().cpu()
            self.best_generator_state = {
                key: value.detach().cpu().clone()
                for key, value in self.model_generator.state_dict().items()
            }

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()

    def _restore_best_generator(self):
        if not hasattr(self, 'best_generator_state'):
            raise RuntimeError('Best generator state is unavailable. Please finish training before exporting weights.')
        self.model_generator.load_state_dict(self.best_generator_state)
        self.model_generator = self.model_generator.to(self.device)
        self.model_generator.eval()

    def _build_ub_weight_filename(self):
        if self.config.contain('trial.signature'):
            return self.config['trial.signature'] + '_ub_weights.txt'
        return self.model_name + '_' + self.datasetname + '_best_epoch_' + str(self.bestPerformance['epoch']) + '_ub_weights.txt'

    def export_training_ub_weights(self):
        if self.bestPerformance is None:
            raise RuntimeError('Best performance is unavailable. Cannot export U-B weights.')

        export_dir = self.output['-dir']
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(export_dir, self._build_ub_weight_filename())

        export_user_emb = self.best_user_emb.to(self.device)
        export_item_emb = self.best_item_emb.to(self.device)
        model_generator = self.model_generator.to(self.device)
        model_generator.eval()

        seen_pairs = set()
        batch_pairs = []
        exported_count = 0

        def flush_batch(handle, pairs):
            nonlocal exported_count
            if not pairs:
                return
            user_indices = torch.as_tensor(
                [self.data.user[user_id] for user_id, _ in pairs],
                device=self.device,
                dtype=torch.long,
            )
            item_indices = torch.as_tensor(
                [self.data.item[item_id] for _, item_id in pairs],
                device=self.device,
                dtype=torch.long,
            )
            with torch.no_grad():
                batch_user_emb = export_user_emb.index_select(0, user_indices)
                batch_item_emb = export_item_emb.index_select(0, item_indices)
                weights = model_generator(batch_user_emb, batch_item_emb).view(-1).detach().cpu().tolist()

            for (user_id, item_id), weight in zip(pairs, weights):
                handle.write(f'{user_id} {item_id} {weight:.6f}\n')
            exported_count += len(pairs)

        with open(export_path, 'w', encoding='utf-8', newline='\n') as handle:
            for user_id, item_id, _ in self.data.training_data:
                pair = (user_id, item_id)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                batch_pairs.append(pair)
                if len(batch_pairs) >= self.export_batch_size:
                    flush_batch(handle, batch_pairs)
                    batch_pairs = []
            flush_batch(handle, batch_pairs)

        print(
            'Exported training U-B weights for best epoch',
            self.bestPerformance['epoch'],
            'to',
            export_path,
            f'({exported_count} edges).'
        )


class Matrix_Factorization(nn.Module):
    def __init__(self, data, emb_size):
        super(Matrix_Factorization, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.embedding_dict = self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        return self.embedding_dict['user_emb'], self.embedding_dict['item_emb']

class NCFEncoder(nn.Module):
    def __init__(self, data, emb_size, mlp_layer, sizes):
        super(NCFEncoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.mlp_layer = mlp_layer
        self.sizes = sizes
        self.in_out = []
        for i in range(len(self.sizes)-1):
            self.in_out.append((self.sizes[i], self.sizes[i+1]))

        self._fc_layers = torch.nn.ModuleList()
        for in_size, out_size in self.in_out:
            self._fc_layers.append(torch.nn.Linear(emb_size*in_size, emb_size*out_size))

        self.embedding_dict = self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_mf_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_mf_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
            'user_mlp_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_mlp_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        mlp_embeddings = torch.cat([self.embedding_dict['user_mlp_emb'], self.embedding_dict['item_mlp_emb']], 0)

        for idx, _ in enumerate(range(len(self._fc_layers))):
            mlp_embeddings = self._fc_layers[idx](mlp_embeddings)
            mlp_embeddings = torch.nn.ReLU()(mlp_embeddings)
        user_mlp_embeddings = mlp_embeddings[:self.data.user_num]
        item_mlp_embeddings = mlp_embeddings[self.data.user_num:]        

        user_embeddings = torch.cat([self.embedding_dict['user_mf_emb'], user_mlp_embeddings], 1)
        item_embeddings = torch.cat([self.embedding_dict['item_mf_emb'], item_mlp_embeddings], 1)

        return user_embeddings, item_embeddings

class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.register_buffer('sparse_norm_adj', rt.TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj))

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings

class NGCF_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(NGCF_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict,self.W = self._init_model()
        self.register_buffer('sparse_norm_adj', rt.TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj))

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        w_dict=dict()
        for i in range(self.layers):
            w_dict['w1_'+str(i)]=nn.Parameter(initializer(torch.empty(self.latent_size, self.latent_size)))
            w_dict['w2_' + str(i)]=nn.Parameter(initializer(torch.empty(self.latent_size, self.latent_size)))
        W = nn.ParameterDict(w_dict)
        return embedding_dict,W

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            temp=torch.mm(ego_embeddings, self.W['w1_' + str(k)])
            ego_embeddings = F.leaky_relu(torch.sparse.mm(self.sparse_norm_adj,temp)+\
                             temp+\
                             torch.mm(torch.sparse.mm(self.sparse_norm_adj,ego_embeddings)*ego_embeddings,self.W['w2_'+str(k)]))
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings


class SimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers):
        super(SimGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.register_buffer('sparse_norm_adj', rt.TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj))

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings)
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings

    def cal_cl_loss(self, idx):
        device = self.sparse_norm_adj.device
        u_idx = torch.unique(torch.tensor(idx[0], dtype=torch.long, device=device))
        i_idx = torch.unique(torch.tensor(idx[1], dtype=torch.long, device=device))
        user_view_1, item_view_1 = self.forward(perturbed=True)
        user_view_2, item_view_2 = self.forward(perturbed=True)
        user_cl_loss = rt.InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = rt.InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)
        return user_cl_loss + item_cl_loss

class SGL_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, n_layers, temp, aug_type):
        super(SGL_Encoder, self).__init__()
        self.data = data
        self.drop_rate = drop_rate
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.temp = temp
        self.aug_type = aug_type
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.register_buffer('sparse_norm_adj', rt.TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj))

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def graph_reconstruction(self):
        if self.aug_type==0 or 1:
            dropped_adj = self.random_graph_augment()
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def random_graph_augment(self):
        dropped_mat = None
        if self.aug_type == 0:
            dropped_mat = rt.GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        elif self.aug_type == 1 or self.aug_type == 2:
            dropped_mat = rt.GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        return rt.TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat, device=self.sparse_norm_adj.device)

    def forward(self, perturbed_adj=None):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            if perturbed_adj is not None:
                if isinstance(perturbed_adj,list):
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings

    def cal_cl_loss(self, idx, perturbed_mat1, perturbed_mat2):
        device = self.sparse_norm_adj.device
        u_idx = torch.unique(torch.tensor(idx[0], dtype=torch.long, device=device))
        i_idx = torch.unique(torch.tensor(idx[1], dtype=torch.long, device=device))
        user_view_1, item_view_1 = self.forward(perturbed_mat1)
        user_view_2, item_view_2 = self.forward(perturbed_mat2)
        view1 = torch.cat((user_view_1[u_idx],item_view_1[i_idx]),0)
        view2 = torch.cat((user_view_2[u_idx],item_view_2[i_idx]),0)
        return rt.InfoNCE(view1, view2, self.temp)


class GraphGenerator_VAE(nn.Module):
    def __init__(self, data, emb_size):
        super(GraphGenerator_VAE, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.encoder = nn.Linear(self.latent_size*2, 64)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc_encoder = nn.Linear(64, 64)
        self.fc_encoder_mu = nn.Linear(64, 16)
        self.fc_encoder_var = nn.Linear(64, 16)
        self.fc_reparameterize = nn.Linear(16, 64)
        self.fc_decode = nn.Linear(64, 1)

    def encode(self, x):
        output = self.encoder(x)
        h = self.relu(output)
        #return self.fc_encoder_mu(h), self.fc_encoder_var(h)
        return self.fc_encoder(h)
 
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.rand_like(std)
        return mu + eps * std
 
    def decode(self, z):
        #h = self.relu(self.fc_reparameterize(z))
        return self.sigmoid(self.fc_decode(z))

    def forward(self, user_e, item_e):
        input_vec = torch.cat((user_e,item_e), axis = 1)
        #input_vec = user_e+item_e
        #mu, log_var = self.encode(input_vec)
        #z = self.reparameterize(mu, log_var)
        z = self.encode(input_vec)
        return self.decode(z)

