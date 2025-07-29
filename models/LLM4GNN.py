from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReplicationPad1d
from torch_geometric.nn import GCNConv

from transformers import (
    LlamaConfig, LlamaModel, LlamaTokenizer,
    GPT2Config, GPT2Model, GPT2Tokenizer,
    BertConfig, BertModel, BertTokenizer
)
import transformers

from layers.StandardNorm import Normalize
from torch.cuda.amp import autocast

# -----------------------------
# 这里导入你刚才定义的两个新模块:
# TemporalPatchEmbedding, TimeTextAligner
# -----------------------------

transformers.logging.set_verbosity_error()

class TimeTextAligner(nn.Module):
    """
    多头交叉注意力，将时序嵌入与文本对齐。
    """
    def __init__(self, d_model, n_heads, d_llm, add_feedforward=True, dropout=0.1):
        super(TimeTextAligner, self).__init__()
        self.add_feedforward = add_feedforward
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, kdim=d_llm, vdim=d_llm, dropout=dropout, batch_first=True)
        if self.add_feedforward:
            self.ffn = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 4*d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4*d_model, d_model),
                nn.Dropout(dropout),
            )
        else:
            self.ffn = None

    def forward(self, ts_embed, text_embed, ts_mask=None, text_mask=None):
        attn_out, _ = self.cross_attn(query=ts_embed, key=text_embed, value=text_embed, key_padding_mask=text_mask, need_weights=False)
        out = ts_embed + attn_out
        if self.ffn is not None:
            out = out + self.ffn(out)
        return out

class FlattenHead(nn.Module):
    """
    输出预测层，将最后的状态拉平后映射到目标预测大小。
    """
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class GNNModel(nn.Module):
    """
    用于时序数据的图神经网络模块。
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class TokenEmbedding(nn.Module):
    """简易的线性投影示例，用于将 patch_len -> d_model"""
    def __init__(self, patch_len, d_model):
        super(TokenEmbedding, self).__init__()
        self.linear = nn.Linear(patch_len, d_model)
        self.linear = self.linear.to(torch.bfloat16)

    def forward(self, x):
        B, num_patches, patch_len = x.shape

        # 强制使用 bfloat16 数据类型，确保一致性
        x = x.to(torch.bfloat16)  # 确保输入数据是 bfloat16
        # reshaped_x = x.reshape(B * num_patches, patch_len)
        return self.linear(x)

class TemporalPatchEmbedding(nn.Module):
    """
    将时序数据分块并映射到 d_model 空间。
    """
    def __init__(self, in_channels, d_model, patch_len, stride, dropout=0.1, use_pos_encoding=True):
        super(TemporalPatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.use_pos_encoding = use_pos_encoding
        self.padding_patch_layer = ReplicationPad1d((0, stride))
        self.value_embedding = TokenEmbedding(patch_len, d_model)
        self.max_num_patches = 5000
        if self.use_pos_encoding:
            self.pos_embedding = nn.Parameter(torch.randn(self.max_num_patches, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, T = x.shape  # B=batch_size, C=channels, T=length of the time series
        n_vars = C

        # Ensure patch_len does not exceed time dimension (T)
        # if self.patch_len > T:
        #     print(f"Warning: patch_len ({self.patch_len}) exceeds time dimension (T={T}). Adjusting patch_len.")
        #     self.patch_len = T  # Adjust patch_len to match the time dimension

        # 确保输入数据是 bfloat16 类型
        x = x.to(torch.bfloat16)  # 强制使用 bfloat16

        # 1) 补边 + unfold
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # 2) 合并 B,C -> [B*C, num_patches, patch_len]
        _, _, num_patches, _ = x.shape
        x = x.reshape(B * C, num_patches, self.patch_len)

        # 3) 投影到 d_model
        x = self.value_embedding(x)  # 将每个patch投影到d_model空间

        # 4) (可选)添加可学习的位置编码
        if self.use_pos_encoding:
            x = x + self.pos_embedding[:num_patches, :]

        x = self.dropout(x)
        return x, n_vars

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.d_model = configs.d_model
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_llm = configs.llm_dim
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len

        # ---------------- 1) 加载 & 冻结 预训练 LLM ----------------
        if configs.llm_model == 'LLAMA':
            llama_config = LlamaConfig.from_pretrained('/home/dragonfly/LLMs/llama-2-7b')
            llama_config.num_hidden_layers = configs.llm_layers
            llama_config.output_attentions = True
            llama_config.output_hidden_states = True
            self.llm_model = LlamaModel.from_pretrained('/home/dragonfly/LLMs/llama-2-7b', config=llama_config)
            self.tokenizer = LlamaTokenizer.from_pretrained('/home/dragonfly/LLMs/llama-2-7b')

        elif configs.llm_model == 'GPT2':
            gpt2_config = GPT2Config.from_pretrained(configs.llm_path)
            self.llm_model = GPT2Model.from_pretrained(configs.llm_path, config=gpt2_config)
            self.tokenizer = GPT2Tokenizer.from_pretrained(configs.llm_path)

        elif configs.llm_model == 'BERT':
            bert_config = BertConfig.from_pretrained(configs.llm_path)
            self.llm_model = BertModel.from_pretrained(configs.llm_path, config=bert_config)
            self.tokenizer = BertTokenizer.from_pretrained(configs.llm_path)
        else:
            raise ValueError("Unsupported LLM model type")

        # 冻结LLM的主干参数
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # 时序数据嵌入层
        self.patch_embedding = TemporalPatchEmbedding(configs.enc_in, self.d_model, self.patch_len, self.stride)

        # GNN部分，处理时序数据的图结构
        # self.gnn_model = GNNModel(in_channels=configs.enc_in, hidden_channels=128, out_channels=self.d_model)
        self.gnn_model = GNNModel(
                                    in_channels=self.d_model,    # ← 32，而不是 enc_in
                                    hidden_channels=128,
                                    out_channels=self.d_model) 

        # 文本对齐模块
        self.time_text_aligner = TimeTextAligner(self.d_model, configs.n_heads, self.d_llm)

        # 线性映射与输出
        self.output_projection = FlattenHead(n_vars=configs.enc_in, nf=self.d_model, target_window=self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 时序数据预处理与图卷积网络
        # 强制转换输入数据类型为 bfloat16，避免混合精度中的类型不一致
        x_enc = x_enc.to(torch.bfloat16)
        x_mark_enc = x_mark_enc.to(torch.bfloat16)
        x_dec = x_dec.to(torch.bfloat16)
        x_mark_dec = x_mark_dec.to(torch.bfloat16)

        enc_out, _ = self.patch_embedding(x_enc)
        x_flat = enc_out.flatten(0, 1) 
        edge_index = self.generate_edge_index(x_flat.size(0), x_flat.device)  # 生成图的边索引（时序依赖关系）
        gnn_out = self.gnn_model(enc_out, edge_index)  # GNN处理

        # 文本与时序数据对齐
        text_embed = self.get_text_embed(x_dec)  # 获取文本嵌入
        aligned_out = self.time_text_aligner(gnn_out, text_embed)

        # 输出预测
        dec_out = self.output_projection(aligned_out)
        return dec_out

    def generate_edge_index(self, num_nodes, device):
        # 这里给一个“时间链”示例：0→1→2→…→N-1，再加自环
        if num_nodes < 2:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        else:
            src = torch.arange(num_nodes-1, device=device)
            dst = src + 1
            edge_index = torch.stack([src, dst], dim=0)       # [2, N-1]
        # 加自环，保证每个节点至少有一条入边
        from torch_geometric.utils import add_self_loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        return edge_index

    def get_text_embed(self, x_dec):
        """
        获取文本嵌入（可以是提示词）。
        """
        prompt_text = "Predict the next steps based on the historical data."
        prompt_enc = self.tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        prompt_ids = prompt_enc.input_ids.to(x_dec.device)
        return self.llm_model.get_input_embeddings()(prompt_ids)  # 使用LLM的嵌入
