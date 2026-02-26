import math
from transformers.modeling_bert import BertModel, BertPreTrainedModel
import torch.nn as nn
import torch
from transformers.modeling_bert import BertIntermediate, BertOutput, BertAttention
from torch.distributions import Normal
from torch.cuda.amp import autocast
import torch.nn.functional as F
from typing import Optional, Tuple

class CrossModalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        assert (
            self.head_dim * self.num_heads == self.hidden_size
        ), "hidden_size 必须能被 num_attention_heads 整除"
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = nn.Dropout(config.dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.scale = self.head_dim ** -0.5
    def forward(
        self,
        query: torch.Tensor,          # [B, L_q, H]
        key_value: torch.Tensor,      # [B, L_kv, H]
    ) -> torch.Tensor:
        residual = query
        B, L_q, H = query.shape
        _, L_kv, _ = key_value.shape
        # 线性映射 + 多头拆分
        Q = self.query(query).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)   # [B, h, L_q, d]
        K = self.key(key_value).view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2) # [B, h, L_kv, d]
        V = self.value(key_value).view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, h, L_q, L_kv]
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        context = torch.matmul(attention_weights, V)  # [B, h, L_q, d]
        context = context.transpose(1, 2).contiguous().view(B, L_q, H)  # [B, L_q, H]
        output = self.output(context)
        output = self.layer_norm(residual + output)
        return output



class MultiScaleFeatureEnhancement(nn.Module):
    """修复版多尺度特征增强模块"""
    def __init__(self, config,hidden_size):
        super().__init__()
        self.hidden_size = hidden_size 
        output_channels = hidden_size // 3 * 3  # 三个卷积分支的输出通道总数
        
        # 多尺度卷积
        self.conv_1x1 = nn.Conv2d(hidden_size, hidden_size//3, 1)
        self.conv_3x3 = nn.Conv2d(hidden_size, hidden_size//3, 3, padding=1)
        self.conv_5x5 = nn.Conv2d(hidden_size, hidden_size//3, 5, padding=2)
        
        # 修复：通道注意力输入通道数改为实际的多尺度输出通道数
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(output_channels, output_channels, 1),  # 输入输出通道数一致
            nn.Sigmoid()
        )
        
        # 修复：融合层输入通道数改为多尺度输出通道数
        self.fusion = nn.Conv2d(output_channels, hidden_size, 1)  # 将多尺度特征映射回原hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        B, L, W, H = x.shape
        x_input = x.permute(0, 3, 1, 2)  # (B, H, L, W)
        
        # 多尺度特征提取
        f1 = self.conv_1x1(x_input)
        f2 = self.conv_3x3(x_input)
        f3 = self.conv_5x5(x_input)
        
        # 特征拼接
        multi_scale = torch.cat([f1, f2, f3], dim=1)  # 通道数: hidden_size//4 * 3
        
        # # 通道注意力
        channel_weights = self.channel_attention(multi_scale)
        attended = multi_scale * channel_weights
        
        # 特征融合：将多尺度特征映射回原维度
        enhanced = self.fusion(attended)  # 输出通道数: hidden_size
        
        enhanced = enhanced.permute(0, 2, 3, 1)  # (B, L, W, H)
        enhanced = self.layer_norm(enhanced)
        
        # 残差连接
        return x + enhanced



# class GraphPropagationReasoning(nn.Module):
    

#     """基于图消息传递的推理 - 遵循图神经网络原理"""  
    
#     def __init__(self, config):
#         super().__init__()
#         self.hidden_size = config.hidden_size
        
#         # 消息函数：基于相邻节点和边特征生成消息
#         self.message_function = nn.Sequential(
#             nn.Linear(3 * config.hidden_size, config.hidden_size),  # 简化输入维度
#             nn.GELU()
#         )
        
#         # 聚合函数：使用注意力机制聚合邻居消息
#         self.attention = nn.MultiheadAttention(
#             embed_dim=config.hidden_size,
#             num_heads=min(4, config.hidden_size // 64),
#             dropout=0.1, 
#         )
        
#         # 更新函数：GRU门控更新节点状态
#         self.update_gate = nn.GRUCell(config.hidden_size, config.hidden_size)
#         self.layer_norm = nn.LayerNorm(config.hidden_size)


#     def forward(self, table_features, sequence_embedding):
#         """
#         基于图传播的推理原理：
#         1. 消息传递：节点间交换信息
#         2. 消息聚合：收集邻居信息  
#         3. 状态更新：根据新信息更新节点状态
#         """
#         B, L, _, H = table_features.shape
        
#         # 使用序列嵌入作为节点特征
#         nodes = sequence_embedding  # [B, L, H]
#         edges = table_features      # [B, L, L, H] 边特征
        
 
#         messages = self.generate_messages(nodes, edges)  # [B, L, L, H]  
        
#         # 步骤2: 消息聚合（注意力机制）
#         aggregated_messages = self.aggregate_messages(nodes, messages)  # [B, L, H]
        
#         # 步骤3: 状态更新（GRU门控）
#         nodes = self.update_node_states(nodes, aggregated_messages)  # [B, L, H]
        
#         # 步骤4: 边特征更新（基于更新后的节点）
#         edges = self.update_edge_features(nodes, edges)  # [B, L, L, H]
        
#         edges = self.layer_norm(edges)
         
#         edges = edges 
#         return edges

#     def generate_messages(self, nodes, edges):
#         """消息生成：基于源节点、目标节点和边特征生成消息"""
#         B, L, H = nodes.shape
        
#         # 源节点特征扩展 [B, L, L, H]
#         source_nodes = nodes.unsqueeze(2).expand(-1, -1, L, -1)
#         # 目标节点特征扩展 [B, L, L, H]  
#         target_nodes = nodes.unsqueeze(1).expand(-1, L, -1, -1)  
        
#         # 组合多种特征
#         combined = torch.cat([
#             source_nodes, edges, target_nodes
#         ], dim=-1)
        
#         # 生成消息
#         messages = self.message_function(combined)
#         return messages

#     def aggregate_messages(self, nodes, messages):
#         """消息聚合：使用注意力机制选择重要消息"""
#         B, L, L, H = messages.shape
        
#         # 重塑为序列形式 [B, L, L, H] -> [B, L*L, H]
#         messages_flat = messages.view(B, L*L, H)
        
#         # 注意力聚合：每个节点关注其收到的所有消息
#         aggregated, _ = self.attention(
#             query=nodes, 
#             key=messages_flat, 
#             value=messages_flat
#         )
        
#         return aggregated  

#     def update_node_states(self, nodes, messages):
#         """状态更新：GRU门控机制更新节点状态"""
#         B, L, H = nodes.shape
        
#         nodes_flat = nodes.reshape(B*L, H)
#         messages_flat = messages.reshape(B*L, H)
        
#         # GRU更新：决定保留多少旧状态，接受多少新信息
#         updated_flat = self.update_gate(messages_flat, nodes_flat)
#         return updated_flat.view(B, L, H)

#     def update_edge_features(self, nodes, edges):
#         """边特征更新：基于更新后的节点重新计算边特征"""
#         B, L, H = nodes.shape
        
#         # 新的边特征 = 源节点 + 原始边特征 + 目标节点
#         source_nodes = nodes.unsqueeze(2).expand(-1  , -1, L, -1)
#         target_nodes = nodes.unsqueeze(1).expand(-1, L, -1, -1)
        
#         # 边特征更新：结合多种信息
#         updated_edges = edges + 0.2 * (source_nodes + target_nodes)
#         return updated_edges



class GraphPropagationReasoning(nn.Module):
    

    """基于图消息传递的推理 - 遵循图神经网络原理"""  
    
    def __init__(self, config,hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 消息函数：基于相邻节点和边特征生成消息
        self.message_function = nn.Sequential(
            nn.Linear(3 * hidden_size,hidden_size),  # 简化输入维度
            nn.GELU()
        )
        
        # 聚合函数：使用注意力机制聚合邻居消息
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=min(4, hidden_size // 64),
            dropout=config.dropout_prob, 
        )
        
        # 更新函数：GRU门控更新节点状态
        self.update_gate = nn.GRUCell(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)


    def forward(self, table_features, sequence_embedding):
        """
        基于图传播的推理原理：
        1. 消息传递：节点间交换信息
        2. 消息聚合：收集邻居信息  
        3. 状态更新：根据新信息更新节点状态
        """
        B, L, _, H = table_features.shape
        
        # 使用序列嵌入作为节点特征
        nodes = sequence_embedding  # [B, L, H]
        edges = table_features      # [B, L, L, H] 边特征
        
 
        messages = self.generate_messages(nodes, edges)  # [B, L, L, H]  
        
        # 步骤2: 消息聚合（注意力机制）
        aggregated_messages = self.aggregate_messages(nodes, messages)  # [B, L, H]
        
        # 步骤3: 状态更新（GRU门控）
        nodes = self.update_node_states(nodes, aggregated_messages)  # [B, L, H]
        
        # 步骤4: 边特征更新（基于更新后的节点）
        edges = self.update_edge_features(nodes, edges)  # [B, L, L, H]
        
        edges = self.layer_norm(edges)
         
        edges = edges 
        return edges

    def generate_messages(self, nodes, edges):
        """消息生成：基于源节点、目标节点和边特征生成消息"""
        B, L, H = nodes.shape
        
        # 源节点特征扩展 [B, L, L, H]
        source_nodes = nodes.unsqueeze(2).expand(-1, -1, L, -1)
        # 目标节点特征扩展 [B, L, L, H]  
        target_nodes = nodes.unsqueeze(1).expand(-1, L, -1, -1)  
        
        # 组合多种特征
        combined = torch.cat([
            source_nodes, edges, target_nodes
        ], dim=-1)
        
        # 生成消息
        messages = self.message_function(combined)
        return messages

    def aggregate_messages(self, nodes, messages):
        """消息聚合：使用注意力机制选择重要消息"""
        B, L, L, H = messages.shape
        
        # 重塑为序列形式 [B, L, L, H] -> [B, L*L, H]
        messages_flat = messages.view(B, L*L, H)
        
        # 注意力聚合：每个节点关注其收到的所有消息
        aggregated, _ = self.attention(
            query=nodes, 
            key=messages_flat, 
            value=messages_flat
        )
        
        return aggregated  

    def update_node_states(self, nodes, messages):
        """状态更新：GRU门控机制更新节点状态"""
        B, L, H = nodes.shape
        
        nodes_flat = nodes.reshape(B*L, H)
        messages_flat = messages.reshape(B*L, H)
        
        # GRU更新：决定保留多少旧状态，接受多少新信息
        updated_flat = self.update_gate(messages_flat, nodes_flat)
        return updated_flat.view(B, L, H)

    def update_edge_features(self, nodes, edges):
        """边特征更新：基于更新后的节点重新计算边特征"""
        B, L, H = nodes.shape
        
        # 新的边特征 = 源节点 + 原始边特征 + 目标节点
        source_nodes = nodes.unsqueeze(2).expand(-1  , -1, L, -1)
        target_nodes = nodes.unsqueeze(1).expand(-1, L, -1, -1)
        
        # 边特征更新：结合多种信息
        updated_edges = edges + 0.2 * (source_nodes + target_nodes)
        return updated_edges
 