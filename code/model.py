from transformers.modeling_bert import BertModel, BertPreTrainedModel
import torch.nn as nn
import torch
from modules import *



class PLGF(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config=config)
        # 固定BERT嵌入层
        if config.fix_bert_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
        self.embd_dropout = nn.Dropout(config.hidden_dropout_prob)
        # 投影层l
        self.hidden_size = config.hidden_size
        self.reasoning_module = GraphPropagationReasoning(config,self.hidden_size // 2)
        self.Lr_sl = nn.Linear(self.hidden_size, self.hidden_size)
        self.Lr_ol = nn.Linear(self.hidden_size, self.hidden_size)
        self.Lr_sr = nn.Linear(self.hidden_size, self.hidden_size)
        self.Lr_or = nn.Linear(self.hidden_size, self.hidden_size)
        self.table_feature_enhance = MultiScaleFeatureEnhancement(config,self.hidden_size // 2)
        self.sl_cross_attention = CrossModalAttention(config)
        self.ol_cross_attention = CrossModalAttention(config)
        self.sr_cross_attention = CrossModalAttention(config)
        self.or_cross_attention = CrossModalAttention(config)    
        self.table_liner = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.tablel_liner = nn.Linear(self.hidden_size * 4, self.hidden_size)
        self.tabler_liner = nn.Linear(self.hidden_size * 4, self.hidden_size)
        # 双路：table_enhanced + table_reasoned 拼接后 [B,L,L,H] -> table_fusion -> [B,L,L,H]
        self.table_fusion = nn.Linear(self.hidden_size, self.hidden_size)
        # 消融时仅剩单路 table [B,L,L,H//2]，用单路投影到 [B,L,L,H]
        self.table_fusion_single = nn.Linear(self.hidden_size // 2, self.hidden_size)
        self.activation = nn.GELU()
        self.embed_linear = nn.Linear(self.hidden_size * 4, self.hidden_size // 2)
        self.enr_label = config.enr_label
        self.en_label = config.en_label
        self.htr_label = config.htr_label
        self.num_p = config.num_p  
        self.en_MLP = nn.Linear(self.hidden_size, config.en_label)
        self.htr_MLP = nn.Linear(self.hidden_size, config.htr_label)  
        self.enr_MLP = nn.Linear(self.hidden_size * 2, config.enr_label * config.num_p)
        self.feature = nn.Linear(config.hidden_size * 6, config.hidden_size * 2)
        # 消融实验开关（从 config 读取，兼容旧 checkpoint）
        self.ablate_cross_attention = getattr(config, 'ablate_cross_attention', False)
        self.ablate_reasoning = getattr(config, 'ablate_reasoning', False)
        self.ablate_table_enhance = getattr(config, 'ablate_table_enhance', False)

    def forward(self, token_ids, mask_token_ids):
        embed = self.get_embed(token_ids, mask_token_ids)
        B, L = embed.shape[0], embed.shape[1]
        sl_ = self.Lr_sl(embed)  
        ol_ = self.Lr_ol(embed)
        sr_ = self.Lr_sr(embed)
        or_ = self.Lr_or(embed)
        if not self.ablate_cross_attention:
            sl_ = self.sl_cross_attention(sl_, torch.cat([ ol_, sr_, or_], dim=1))
            ol_ = self.ol_cross_attention(ol_, torch.cat([ sl_, sr_, or_], dim=1))
            sr_ = self.sr_cross_attention(sr_, torch.cat([ sl_, ol_, or_], dim=1))
            or_ = self.or_cross_attention(or_, torch.cat([ sl_, ol_, sr_], dim=1))
        feature = torch.cat([or_, sl_, ol_, sr_], dim=-1)
        tablel = self.tablel_liner(feature)
        tabler = self.tabler_liner(feature) 
        embed_= self.embed_linear(feature) 
        table = self.activation(tablel.unsqueeze(2) * tabler.unsqueeze(1))
        table = self.table_liner(table)
        table_enhanced = table if self.ablate_table_enhance else self.table_feature_enhance(table)
        table_reasoned = table if self.ablate_reasoning else self.reasoning_module(table, embed_)
        # 维度：table_enhanced / table_reasoned 均为 [B,L,L, hidden_size//2]
        if self.ablate_table_enhance and self.ablate_reasoning:
            # 两路都消融时只有单路 table，用单路投影到 hidden_size，避免 concat(table, table) 的冗余
            tableall = self.table_fusion_single(table)
        else:
            tableall = self.table_fusion(torch.cat([table_enhanced, table_reasoned], dim=-1))
        enr_feature = self.extract_enhanced_features(tableall)
        en_table = self.en_MLP(tableall) 
        htr_table = self.htr_MLP(tableall) 
        enr_table = self.enr_MLP(enr_feature)  
            
        return (enr_table.reshape(B, L, self.num_p, self.enr_label), 
                en_table.reshape(B, L, L, self.en_label),  
                htr_table.reshape(B, L, L, self.htr_label))
    
    def get_embed(self, token_ids, mask_token_ids):
        """获取BERT嵌入"""  
        bert_out = self.bert(
            input_ids=token_ids.long(), 
            attention_mask=mask_token_ids.long()
        )
        return self.embd_dropout(bert_out[0]) 
     
    def extract_enhanced_features(self, tableall):
        """
        提取行和列的特征：标准差、最大值、平均值、对角线特征
        使用简单的加权求和
        
        tableall: [B, L, L, H]
        返回: e1_ [B, L, H], e2_ [B, L, H]
        """
        B, L, _, H = tableall.shape 
        
        # 提取行特征
        e1_mean = tableall.mean(dim=2)
        e1_max, _ = tableall.max(dim=2)
        e1_std = tableall.std(dim=2)
        # 提取列特征
        e2_mean = tableall.mean(dim=1)
        e2_max, _ = tableall.max(dim=1)
        e2_std = tableall.std(dim=1)
        # 组合特征
        features = torch.cat([e1_mean, e1_max, e1_std,e2_mean, e2_max, e2_std], dim=-1)
        # 线性变换
        features = self.feature(features)

        
        return features
