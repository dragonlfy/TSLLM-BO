import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReplicationPad1d
import torch_geometric.nn as gnn

from transformers import (
    LlamaConfig, LlamaModel, LlamaTokenizer,
    GPT2Config, GPT2Model, GPT2Tokenizer,
    BertConfig, BertModel, BertTokenizer
)
import transformers

from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()

class TokenEmbedding(nn.Module):
    def __init__(self, patch_len, d_model):
        super(TokenEmbedding, self).__init__()
        self.linear = nn.Linear(patch_len, d_model)

    def forward(self, x):
        return self.linear(x)

class TemporalPatchEmbedding(nn.Module):
    def __init__(self, 
                 in_channels,   
                 d_model, 
                 patch_len, 
                 stride, 
                 dropout=0.1, 
                 use_pos_encoding=True):
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
        B, C, T = x.shape
        n_vars = C

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        _, _, num_patches, _ = x.shape
        x = x.reshape(B * C, num_patches, self.patch_len)
        x = self.value_embedding(x)
        
        if self.use_pos_encoding:
            x = x + self.pos_embedding[:num_patches, :]

        x = self.dropout(x)
        return x, n_vars

class GraphNeuralNetwork(nn.Module):
    """
    GNN模块用于建模通道间的空间依赖关系
    - 输入: [B*C, num_patches, d_model]
    - 输出: [B*C, num_patches, d_model] (增强空间信息)
    """
    def __init__(self, d_model, num_nodes, gnn_layers=2, gnn_type="gcn", dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.gnn_type = gnn_type
        self.gnn_layers = gnn_layers
        
        # 自适应邻接矩阵学习
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, d_model))
        self.edge_learner = nn.Sequential(
            nn.Linear(2 * d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, 1)
        )
        
        # GNN层选择
        self.gnn_layers = nn.ModuleList()
        for _ in range(gnn_layers):
            if gnn_type == "gcn":
                self.gnn_layers.append(gnn.GCNConv(d_model, d_model))
            elif gnn_type == "gat":
                self.gnn_layers.append(gnn.GATConv(d_model, d_model, heads=1))
            elif gnn_type == "graphsage":
                self.gnn_layers.append(gnn.SAGEConv(d_model, d_model))
            elif gnn_type == "gin":
                self.gnn_layers.append(gnn.GINConv(
                    nn.Sequential(
                        nn.Linear(d_model, 2 * d_model),
                        nn.ReLU(),
                        nn.Linear(2 * d_model, d_model)
                )))
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
                
        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(gnn_layers)])
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, n_vars):
        """
        x: [B*C, num_patches, d_model]
        n_vars: 原始通道数 (C)
        返回: [B*C, num_patches, d_model] (增强空间信息)
        """
        B_C, num_patches, d_model = x.shape
        B = B_C // n_vars
        
        # 1. 通道特征聚合 (每个通道取平均)
        channel_features = x.view(B, n_vars, num_patches, d_model).mean(dim=2)  # [B, C, d_model]
        
        # 2. 构建自适应邻接矩阵
        src_emb = self.node_embeddings.unsqueeze(0).repeat(B, 1, 1)  # [B, C, d_model]
        dst_emb = self.node_embeddings.unsqueeze(0).repeat(B, 1, 1)
        
        # 计算节点对特征
        src_emb_exp = src_emb.unsqueeze(2).repeat(1, 1, n_vars, 1)  # [B, C, C, d_model]
        dst_emb_exp = dst_emb.unsqueeze(1).repeat(1, n_vars, 1, 1)  # [B, C, C, d_model]
        # src_emb_exp = src_emb.unsqueeze(2).expand(B, n_vars, n_vars, d_model)
        # dst_emb_exp = dst_emb.unsqueeze(1).expand(B, n_vars, n_vars, d_model)
        
        # 计算边权重
        # edge_features = torch.cat([src_emb_exp, dst_emb_exp], dim=-1)
        edge_features = torch.cat([src_emb, dst_emb], dim=-1)
        adj_logits = self.edge_learner(edge_features).squeeze(-1)  # [B, C, C]
        
        # 创建稀疏邻接矩阵
        adj_matrix = F.softmax(adj_logits, dim=-1)  # [B, C, C]
        
        # 3. 应用GNN层
        for i in range(self.gnn_layers):
            # 图卷积
            if self.gnn_type in ["gcn", "gat", "graphsage"]:
                channel_features = self.gnn_layers[i](channel_features, adj_matrix)
            else:  # GIN
                channel_features = self.gnn_layers[i](channel_features, adj_matrix)
            
            # 归一化 + 激活 + dropout
            channel_features = self.norm_layers[i](channel_features)
            channel_features = self.activation(channel_features)
            channel_features = self.dropout(channel_features)
        
        # 4. 将通道特征广播回所有patch
        channel_features = channel_features.unsqueeze(2)  # [B, C, 1, d_model]
        # channel_features = channel_features.repeat(1, 1, num_patches, 1)  # [B, C, num_patches, d_model]
        channel_features = channel_features.expand(-1, -1, num_patches, -1)
        channel_features = channel_features.reshape(B_C, num_patches, d_model)  # [B*C, num_patches, d_model]
        
        # 5. 残差连接
        x = x + channel_features
        return x

class TimeTextAligner(nn.Module):
    def __init__(self, d_model, n_heads, d_llm, add_feedforward=True, dropout=0.1):
        super(TimeTextAligner, self).__init__()
        self.add_feedforward = add_feedforward

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            kdim=d_llm,
            vdim=d_llm,
            dropout=dropout,
            batch_first=True
        )

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
        attn_out, _ = self.cross_attn(
            query=ts_embed,
            key=text_embed,
            value=text_embed,
            key_padding_mask=text_mask,
            need_weights=False
        )
        out = ts_embed + attn_out

        if self.ffn is not None:
            out = out + self.ffn(out)

        return out

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.enc_in = configs.enc_in  # 通道数

        # ---------------- 1) 加载 & 冻结 预训练 LLM ----------------
        if configs.llm_model == 'LLAMA':
            llama_config = LlamaConfig.from_pretrained('/home/dragonfly/LLMs/llama-2-7b')
            llama_config.num_hidden_layers = configs.llm_layers
            llama_config.output_attentions = True
            llama_config.output_hidden_states = True

            try:
                self.llm_model = LlamaModel.from_pretrained(
                    '/home/dragonfly/LLMs/llama-2-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=llama_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download from HF...")
                self.llm_model = LlamaModel.from_pretrained(
                    '/home/dragonfly/LLMs/llama-2-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=llama_config,
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    '/home/dragonfly/LLMs/llama-2-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download...")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    '/home/dragonfly/LLMs/llama-2-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )

        elif configs.llm_model == 'GPT2':
            gpt2_config = GPT2Config.from_pretrained(configs.llm_path)
            gpt2_config.num_hidden_layers = configs.llm_layers
            gpt2_config.output_attentions = True
            gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=gpt2_config,
                )
            except EnvironmentError:
                print("Local GPT2 model not found. Attempting to download from HF...")
                self.llm_model = GPT2Model.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=gpt2_config,
                )
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local GPT2 tokenizer not found. Attempting to download from HF...")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=False
                )

        elif configs.llm_model == 'BERT':
            bert_config = BertConfig.from_pretrained(configs.llm_path)
            bert_config.num_hidden_layers = configs.llm_layers
            bert_config.output_attentions = True
            bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=bert_config,
                )
            except EnvironmentError:
                print("Local BERT model not found. Attempting to download from HF...")
                self.llm_model = BertModel.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=bert_config,
                )
            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local BERT tokenizer not found. Attempting to download from HF...")
                self.tokenizer = BertTokenizer.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise ValueError("Unsupported LLM model type")

        # 处理 tokenizer 的 pad_token
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # 冻结 LLM 主干参数
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # 一些模型外部配置
        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = (
                "This is a time-series forecasting scenario. "
                "We use a chain-of-thought style to reason step by step."
            )

        self.dropout = nn.Dropout(configs.dropout)

        # 2) 使用改进后的 TemporalPatchEmbedding
        self.patch_embedding = TemporalPatchEmbedding(
            in_channels=configs.enc_in,
            d_model=configs.d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            dropout=configs.dropout,
            use_pos_encoding=True
        )

        # 3) 添加GNN模块 (新组件)
        self.gnn = GraphNeuralNetwork(
            d_model=configs.d_model,
            num_nodes=configs.enc_in,  # 节点数=通道数
            gnn_layers=configs.gnn_layers,  # 新增配置参数
            gnn_type=configs.gnn_type,      # 新增配置参数
            # gnn_layers=configs.gnn_layers,  # 新增配置参数
            # gnn_type=configs.gnn_type,      # 新增配置参数
            dropout=configs.dropout
        )

        # 源词向量（LLM 原本的 embedding），后面做映射
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]

        # 可训练的 mapping 层
        self.num_tokens = 100
        self.num_prototypes = 100
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.d_model = configs.d_model
        self.text_prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.d_llm))
        
        # 4) 使用改进后的 TimeTextAligner 做跨模态对齐
        self.time_text_aligner = TimeTextAligner(
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_llm=self.d_llm,
            add_feedforward=True,
            dropout=configs.dropout
        )

        # 计算 patch 的数量
        self.patch_nums = int((self.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = configs.d_ff * self.patch_nums

        # 输出层
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.output_projection = FlattenHead(
                n_vars=configs.enc_in,
                nf=self.head_nf,
                target_window=self.pred_len,
                head_dropout=configs.dropout
            )
        else:
            raise NotImplementedError(
                f"Task {self.task_name} not implemented for this model."
            )

        # 实例归一化层
        self.normalize_layers = Normalize(configs.enc_in, affine=False)

        # 其他
        self.top_k = 5

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        else:
            return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 1) Normalization
        x_enc = self.normalize_layers(x_enc, 'norm')  # [B, T, N]
        B, T, N = x_enc.size()

        # 转为 [B, N, T] 以适配 TemporalPatchEmbedding
        x_enc = x_enc.permute(0, 2, 1).contiguous()  # [B, N, T]

        # 构造提示
        x_stat = x_enc.reshape(B*N, T, 1)
        min_values = torch.min(x_stat, dim=1)[0]
        max_values = torch.max(x_stat, dim=1)[0]
        medians = torch.median(x_stat, dim=1).values
        lags = self.calculate_lags(x_stat)
        trends = x_stat.diff(dim=1).sum(dim=1)

        # prompt_text [保持原有代码不变] ...
        prompt_text = []
        for b in range(x_stat.size(0)):
            min_v = str(min_values[b].tolist()[0])
            max_v = str(max_values[b].tolist()[0])
            median_v = str(medians[b].tolist()[0])
            lags_v = str(lags[b].tolist())
            up_or_down = 'upward' if trends[b] > 0 else 'downward'

            prompt_ = (
                f"<|start_prompt|>Data Domain: {self.description}\n"
                f"Task: Predict next {self.pred_len} steps from last {self.seq_len} steps.\n"
                f"Chain-of-thought: Let's reason step by step.\n\n"
                f"[Input Statistics]\n"
                f" - min value: {min_v}\n"
                f" - max value: {max_v}\n"
                f" - median value: {median_v}\n"
                f" - overall trend: {up_or_down}\n"
                f" - top 5 lags: {lags_v}\n"
                f"<|end_prompt|>\n"
            )
            prompt_text.append(prompt_)

        prompt_enc = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        prompt_ids = prompt_enc.input_ids.to(x_enc.device)  # [B*N, prompt_len]
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_ids)

        # 2) Patch Embedding (多通道)
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))

        # 3) 应用GNN (新添加的步骤)
        enc_out = self.gnn(enc_out, n_vars)

        # 4) TimeTextAligner = Cross-Attn
        source_proj = self.mapping_layer(self.word_embeddings.permute(1, 0))
        self.extra_llm_map = nn.Linear(self.num_tokens, self.d_llm, bias=True).to(torch.bfloat16).to(x_enc.device)
        text_embed = self.extra_llm_map(source_proj) 
        enc_out = self.time_text_aligner(
            ts_embed=enc_out,            # [B*N, patch_nums, d_model]
            text_embed=text_embed        # [B*N, vocab_size, d_llm]
        )
        self.enc_proj = nn.Linear(self.d_model, self.d_llm).to(torch.bfloat16).to(x_enc.device)

        # 在使用时
        # enc_out = self.enc_proj(enc_out) 
        enc_out = self.enc_proj(enc_out) if hasattr(self, 'enc_proj') else enc_out

        # 5) 拼上 prompt_embeddings
        combined_emb = torch.cat([prompt_embeddings, enc_out], dim=1)
        
        # 6) 送入冻结的 LLM
        llm_output = self.llm_model(inputs_embeds=combined_emb)
        dec_out = llm_output.last_hidden_state 
        
        # 7) 恢复形状 -> FlattenHead
        dec_out = dec_out[:, -enc_out.shape[1]:, :self.d_ff]  # 仅取 patch部分 & d_ff
        dec_out = dec_out.reshape(-1, n_vars, dec_out.shape[1], dec_out.shape[2])  # [B, N, patch_nums, d_ff]
        dec_out = dec_out.permute(0, 1, 3, 2)  # [B, N, d_ff, patch_nums]
        # print(f"[DEBUG] dec_out shape before output_projection: {dec_out.shape}")
        dec_out = self.output_projection(dec_out)
        # print(f"[DEBUG] dec_out shape after output_projection: {dec_out.shape}")
        dec_out = dec_out.permute(0, 2, 1) 
        
        # 8) 反归一化
        dec_out = self.normalize_layers(dec_out, 'denorm')
        return dec_out

    def calculate_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags