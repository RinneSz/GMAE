from lr import PolynomialDecayLR
import torch
import math
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl

from torch.nn import functional as F


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class GMAE_node(pl.LightningModule):
    def __init__(
        self,
        n_encoder_layers,
        n_decoder_layers,
        num_heads,
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
        attention_dropout_rate,
        n_node_features,
        mask_ratio,
        n_val_sampler,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_node_features = n_node_features
        self.n_val_sampler = n_val_sampler
        self.mask_ratio = mask_ratio
        self.num_heads = num_heads
        self.input_proj = nn.Linear(n_node_features, hidden_dim)

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

        self.out_proj = nn.Linear(hidden_dim, self.n_node_features)
        self.loss_fn = F.mse_loss

        self.dataset_name = dataset_name

        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay

        self.hidden_dim = hidden_dim
        self.automatic_optimization = True
        self.apply(lambda module: init_params(module, n_layers=n_encoder_layers))

    def compute_pos_embeddings(self, batched_data):
        attn_bias, spatial_pos, x = batched_data.attn_bias, batched_data.spatial_pos, batched_data.x
        in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
        # graph_attn_bias
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node, n_node]
        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias = graph_attn_bias + spatial_pos_bias
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        node_feature = self.input_proj(x)
        node_feature = node_feature + \
            self.in_degree_encoder(in_degree) + \
            self.out_degree_encoder(out_degree)
        graph_node_feature = node_feature

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
        output = self.out_proj(output)  # [n_graph, n_node, n_feature]
        return output

    def forward(self, batched_data, mask=None):
        graph_node_feature, graph_attn_bias = self.compute_pos_embeddings(batched_data)
        in_degree = batched_data.in_degree
        out_degree = batched_data.out_degree

        # compute graph_mask that masks graphs with invalid values
        graph_mask = None
        if mask is not None:
            graph_attn_bias_masked = graph_attn_bias[:, :, ~mask, :][:, :, :, ~mask]
            if graph_attn_bias_masked.size(3) == torch.isinf(graph_attn_bias_masked).sum(3).max().item():
                n_graph = graph_attn_bias_masked.size(0)
                sup = graph_attn_bias_masked.reshape(n_graph, -1)
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
        output = output.reshape(-1, self.n_val_sampler, output.size(1), self.hidden_dim)[:, :, 0, :].mean(1)
        output = output  # [n_graph(n_central_node), n_feature]
        return output

    def training_step(self, batched_data, batch_idx):
        num_nodes = batched_data.x.size(1)
        # do not mask the global nodes
        num_mask = int(self.mask_ratio * num_nodes)
        mask = np.hstack([
            np.zeros(num_nodes - num_mask),
            np.ones(num_mask),
        ])
        np.random.shuffle(mask)
        mask = torch.Tensor(mask).bool()
        y_hat, graph_mask = self(batched_data, mask)  # [n_graph, n_masked_node, n_feature]
        if graph_mask is not None:
            y_gt = batched_data.x[graph_mask][:, mask].float()
        else:
            y_gt = batched_data.x[:, mask].float()
        y_hat = y_hat.reshape(-1, y_hat.size(2))  # [n_graph*n_masked_node, n_feature]
        y_gt = y_gt.reshape(-1, y_gt.size(2))  # [n_graph*n_masked_node, n_feature]
        pad_mask = torch.nonzero(y_gt.sum(-1))
        y_gt = y_gt[pad_mask, :].reshape(-1)
        y_hat = y_hat[pad_mask, :].reshape(-1)
        loss = self.loss_fn(y_hat, y_gt)
        self.log('train_loss', loss)

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
        parser = parent_parser.add_argument_group("GMAE_node")
        parser.add_argument('--n_encoder_layers', type=int, default=3)
        parser.add_argument('--n_decoder_layers', type=int, default=3)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--hidden_dim', type=int, default=64)
        parser.add_argument('--ffn_dim', type=int, default=64)
        parser.add_argument('--intput_dropout_rate', type=float, default=0.1)
        parser.add_argument('--dropout_rate', type=float, default=0.5)
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--attention_dropout_rate',type=float, default=0.1)
        parser.add_argument('--checkpoint_path', type=str, default='')
        parser.add_argument('--warmup_updates', type=int, default=40000)
        parser.add_argument('--tot_updates', type=int, default=400000)
        parser.add_argument('--peak_lr', type=float, default=0.0001)
        parser.add_argument('--end_lr', type=float, default=1e-9)
        parser.add_argument('--mask_ratio', type=float, default=0.5)
        parser.add_argument('--validate', action='store_true', default=False)
        parser.add_argument('--test', action='store_true', default=False)
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
