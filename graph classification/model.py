from data import get_dataset
from lr import PolynomialDecayLR
import torch
import math
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class GMAE_graph(pl.LightningModule):
    def __init__(
        self,
        n_encoder_layers,
        n_decoder_layers,
        num_heads,
        n_node_features,
        n_edge_features,
        hidden_dim,
        dropout_rate,
        intput_dropout_rate,
        weight_decay,
        ffn_dim,
        dataset_name,
        warmup_updates,
        tot_updates,
        peak_lr,
        end_lr,
        edge_type,
        multi_hop_max_dist,
        attention_dropout_rate,
        mask_ratio,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.mask_ratio = mask_ratio
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.num_heads = num_heads

        self.atom_encoder = nn.Embedding(
            512 * self.n_node_features + 1, hidden_dim, padding_idx=0)
        if self.n_edge_features is not None:
            self.edge_encoder = nn.Embedding(
                512 * self.n_edge_features + 1, num_heads, padding_idx=0)
        self.edge_type = edge_type
        if self.edge_type == 'multi_hop':
            self.edge_dis_encoder = nn.Embedding(
                128 * num_heads * num_heads, 1)
        self.spatial_pos_encoder = nn.Embedding(512, num_heads, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(
            512, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(
            512, hidden_dim, padding_idx=0)

        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads)
                    for _ in range(n_encoder_layers)]
        self.encoder_layers = nn.ModuleList(encoders)
        self.encoder_final_ln = nn.LayerNorm(hidden_dim)

        self.encoder_to_decoder = nn.Linear(hidden_dim, hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        decoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads)
                    for _ in range(n_decoder_layers)]
        self.decoder_layers = nn.ModuleList(decoders)
        self.decoder_final_ln = nn.LayerNorm(hidden_dim)

        if dataset_name == 'ZINC':
            self.out_proj = nn.Linear(hidden_dim, 21)  # num_features=1, ranges from 0 to 20, i.e. 21 classes
        else:
            self.out_proj = nn.Linear(hidden_dim, self.n_node_features)

        self.graph_token = nn.Embedding(1, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        if dataset_name == 'ZINC':
            self.loss_fn = nn.NLLLoss()
        else:
            self.loss_fn = get_dataset(dataset_name)['loss_fn']

        self.dataset_name = dataset_name

        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.multi_hop_max_dist = multi_hop_max_dist

        self.hidden_dim = hidden_dim
        self.automatic_optimization = True
        self.apply(lambda module: init_params(module, n_layers=n_encoder_layers))

    def compute_pos_embeddings(self, batched_data):
        attn_bias, spatial_pos, x = batched_data.attn_bias.cuda(), batched_data.spatial_pos.cuda(), batched_data.x.cuda()
        in_degree, out_degree = batched_data.in_degree.cuda(), batched_data.in_degree.cuda()
        # graph_attn_bias
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node, n_node]
        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias = graph_attn_bias + spatial_pos_bias

        if batched_data.edge_input is not None:
            edge_input, attn_edge_type = batched_data.edge_input.cuda(), batched_data.attn_edge_type.cuda()
            # edge feature
            if self.edge_type == 'multi_hop':
                spatial_pos_ = spatial_pos.clone()
                spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
                # set 1 to 1, x > 1 to x - 1
                spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
                if self.multi_hop_max_dist > 0:
                    spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                    edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
                # [n_graph, n_node, n_node, max_dist, n_head]
                edge_input = self.edge_encoder(edge_input).mean(-2)
                max_dist = edge_input.size(-2)
                edge_input_flat = edge_input.permute(
                    3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
                edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads)[:max_dist, :, :])
                edge_input = edge_input_flat.reshape(
                    max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
                edge_input = (edge_input.sum(-2) /
                              (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
            else:
                # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
                edge_input = self.edge_encoder(
                    attn_edge_type).mean(-2).permute(0, 3, 1, 2)
            graph_attn_bias = graph_attn_bias + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        node_feature = self.atom_encoder(x.long()).sum(dim=-2)  # [n_graph, n_node, n_hidden]

        graph_node_feature = node_feature + \
            self.in_degree_encoder(in_degree) + \
            self.out_degree_encoder(out_degree)

        return graph_node_feature, graph_attn_bias

    def encoder(self, graph_node_feature, graph_attn_bias, mask=None):
        if mask is not None:
            graph_node_feature_masked = graph_node_feature[:, ~mask]  # [n graph, n non-masked nodes, n hidden]
            graph_attn_bias_masked = graph_attn_bias[:, :, ~mask, :][:, :, :, ~mask]  # [n graph, n heads, n non-masked nodes, n non-masked nodes]
        else:
            graph_node_feature_masked = graph_node_feature
            graph_attn_bias_masked = graph_attn_bias
        # transfomrer encoder
        output = self.input_dropout(graph_node_feature_masked)
        for enc_layer in self.encoder_layers:
            output = enc_layer(output, graph_attn_bias_masked)
        output = self.encoder_final_ln(output)
        return output

    def decoder(self, output, in_degree, out_degree, graph_attn_bias, mask=None):
        if mask is not None:
            pos_embed = self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
            pos_embed_vis = pos_embed[:, ~mask]
            pos_embed_mask = pos_embed[:, mask]
            node_index_mask = mask.nonzero().view(-1)
            node_index_vis = (~mask).nonzero().view(-1)
            new_node_index = torch.cat([node_index_vis, node_index_mask])
            graph_attn_bias = graph_attn_bias[:, :, new_node_index][:, :, :, new_node_index]
            output = torch.cat([output + pos_embed_vis, self.mask_token + pos_embed_mask], dim=1)
            num_masked = pos_embed_mask.shape[1]
        for enc_layer in self.decoder_layers:
            output = enc_layer(output, graph_attn_bias)
        if mask is not None:
            output = self.decoder_final_ln(output[:, -num_masked:])
        else:
            output = self.decoder_final_ln(output)

        # output part
        output = self.out_proj(output)
        if self.dataset_name == 'ZINC':
            output = torch.log_softmax(output, dim=-1)  # for NLL loss
        return output

    def forward(self, batched_data, mask=None):
        graph_node_feature, graph_attn_bias = self.compute_pos_embeddings(batched_data)
        in_degree = batched_data.in_degree
        out_degree = batched_data.out_degree

        graph_mask = None
        if mask is not None:
            graph_attn_bias_masked = graph_attn_bias[:, :, ~mask, :][:, :, :, ~mask]
            if graph_attn_bias_masked.size(3) == torch.isinf(graph_attn_bias_masked).sum(3).max().item():
                n_graph = graph_attn_bias_masked.size(0)
                sup = graph_attn_bias_masked.view(n_graph, -1)
                length = sup.size(1)
                infs = torch.isinf(sup).sum(1)
                graph_mask = ~(infs == length).bool()
                graph_node_feature = graph_node_feature[graph_mask]
                graph_attn_bias = graph_attn_bias[graph_mask]
                in_degree = in_degree[graph_mask]
                out_degree = out_degree[graph_mask]
        output = self.encoder(graph_node_feature, graph_attn_bias, mask)
        output = self.encoder_to_decoder(output)
        output = self.decoder(output, in_degree, out_degree, graph_attn_bias, mask)
        return output, graph_mask

    def generate_pretrain_embeddings_for_downstream_task(self, batched_data):
        graph_node_feature, graph_attn_bias = self.compute_pos_embeddings(batched_data)
        output = self.encoder(graph_node_feature, graph_attn_bias)
        return output

    def training_step(self, batched_data, batch_idx):
        num_nodes = batched_data.x.size(1)
        num_mask = int(self.mask_ratio * num_nodes)
        mask = np.hstack([
            np.zeros(num_nodes - num_mask),
            np.ones(num_mask),
        ])
        np.random.shuffle(mask)
        mask = torch.Tensor(mask).bool()

        if self.dataset_name == 'ZINC':
            y_hat, graph_mask = self(batched_data, mask)
            y_hat = y_hat.view(-1, 21)  # [n graphs, n nodes, n feature classes=21]->[n graphs * n nodes, 21]
            if graph_mask is not None:
                y_gt = batched_data.x[graph_mask][:, mask].long().view(-1)
            else:
                y_gt = batched_data.x[:, mask].long().view(-1)
            mask = torch.nonzero(y_gt)
            y_hat = y_hat[mask, :].view(-1, 21)
            y_gt = y_gt[mask]-2  # minus 2 since convert_to_single_embed and pad_2d_unsqueeze added 2 in total
            y_gt = y_gt % 512
            y_gt = y_gt.view(-1)
        else:
            y_hat, graph_mask = self(batched_data, mask)
            if graph_mask is not None:
                y_gt = batched_data.x[graph_mask][:, mask].float().view(-1)
            else:
                y_gt = batched_data.x[:, mask].float().view(-1)
            y_hat = y_hat.view(-1)  # [n graphs, n nodes, n feature classes=3]->[n graphs * n nodes, 3]
            mask = torch.nonzero(y_gt)
            y_hat = y_hat[mask].view(-1)
            y_gt = y_gt[mask].view(-1)
            y_gt = y_gt - 2  # minus 2 since convert_to_single_embed and pad_2d_unsqueeze added 2 in total
            y_gt = y_gt % 512
        loss = self.loss_fn(y_hat, y_gt)

        self.log('train_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay)
        lr_scheduler = {
            'scheduler': PolynomialDecayLR(
                optimizer,
                warmup_updates=self.warmup_updates,
                tot_updates=self.tot_updates,
                lr=self.peak_lr,
                end_lr=self.end_lr,
                power=1.0,
            ),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer], [lr_scheduler]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GMAE_graph")
        parser.add_argument('--n_encoder_layers', type=int, default=12)
        parser.add_argument('--n_decoder_layers', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=32)
        parser.add_argument('--hidden_dim', type=int, default=512)
        parser.add_argument('--ffn_dim', type=int, default=512)
        parser.add_argument('--intput_dropout_rate', type=float, default=0.1)
        parser.add_argument('--dropout_rate', type=float, default=0.1)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--attention_dropout_rate', type=float, default=0.1)
        parser.add_argument('--checkpoint_path', type=str, default='')
        parser.add_argument('--warmup_updates', type=int, default=60000)
        parser.add_argument('--tot_updates', type=int, default=1000000)
        parser.add_argument('--peak_lr', type=float, default=2e-4)
        parser.add_argument('--end_lr', type=float, default=1e-9)
        parser.add_argument('--edge_type', type=str, default='multi_hop')
        parser.add_argument('--validate', action='store_true', default=False)
        parser.add_argument('--test', action='store_true', default=False)
        parser.add_argument('--mask_ratio', type=float, default=0.5)
        parser.add_argument('--early_stop_epoch', type=int, default=50)
        return parent_parser


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=3)
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
