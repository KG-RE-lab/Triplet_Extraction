import os
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, BertConfig
from model import PLGF
from util import *
import logging
import torch.nn.functional as F
class UnifiedLogger:
    """统一的日志系统，所有日志写入同一个文件，包含训练性能指标"""
    def __init__(self, log_file):
        self.log_file = log_file
        
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else "."
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # 创建logger
        logger_name = f"UnifiedLogger_{log_file}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        # 禁用传播到根logger
        self.logger.propagate = False
        
        # 清除已有的handler（防止重复）
        self.logger.handlers.clear()
        
        # 文件handler - 覆盖模式写入（每次运行覆盖旧日志，不输出到控制台）
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(message)s', 
                                         datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, message):
        """记录INFO级别日志"""
        self.logger.info(message)
    
    def warning(self, message):
        """记录WARNING级别日志"""
        self.logger.warning(message)
    
    def error(self, message):
        """记录ERROR级别日志"""
        self.logger.error(message)
def setup_environment(args):
    """设置运行环境"""
    try:
        torch.cuda.set_device(int(args.cuda_id))
    except:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id

def load_config_and_model(args, id2predicate):
    """加载配置和模型"""
    config = BertConfig.from_pretrained(args.bert_config_path)
    config.num_p = len(id2predicate)
    config.fix_bert_embeddings = args.fix_bert_embeddings
    config.enr_label = 4 # 主体和关系的相互，0，1
    config.en_label = 2   # 是否是实体
    config.htr_label = 6   # 头尾的相互，0:null，1:SH,HS,H，2:T，3:S
    config.dropout_prob = 0.1
    # 消融实验：从 --ablate 列表解析要关闭的模块（可消融一个、两个或三个）
    ablate_list = getattr(args, 'ablate', []) or []
    config.ablate_cross_attention = 'cross_attention' in ablate_list
    config.ablate_reasoning = 'reasoning' in ablate_list
    config.ablate_table_enhance = 'table_enhance' in ablate_list

    model = CAMGT.from_pretrained(
        pretrained_model_name_or_path=args.bert_model_path, 
        config=config
    )
    model.to("cuda")
    return model


class CrossEntropyLossWithMask(nn.Module):
    """带 mask 的交叉熵损失，接口与 FocalLoss_plus 一致：forward(pred, target, mask=None)"""
    def __init__(self, alpha=None, reduction='none'):
        super(CrossEntropyLossWithMask, self).__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float32) if alpha is not None else None
        self.reduction = reduction

    def forward(self, pred, target, mask=None):
        pred = pred.float()
        target = target.long()
        if mask is not None:
            mask = mask.float()
        original_shape = pred.shape
        num_classes = original_shape[-1]
        pred = pred.reshape(-1, num_classes)
        target = target.reshape(-1)
        weight = None
        if self.alpha is not None:
            if self.alpha.device != pred.device:
                self.alpha = self.alpha.to(pred.device)
            weight = self.alpha
        loss = F.cross_entropy(pred, target, reduction='none', weight=weight)
        if mask is not None:
            mask = mask.reshape(-1)
            loss = loss * mask
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        return loss


class FocalLoss(nn.Module):
    """标准 Focal Loss: -alpha * (1 - p_t)^gamma * log(p_t)"""
    def __init__(self, alpha=None, gamma=2.0, reduction='none'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha, dtype=torch.float32) if alpha is not None else None
        self.reduction = reduction

    def forward(self, pred, target, mask=None):
        pred = pred.float()
        target = target.long()
        if mask is not None:
            mask = mask.float()
        original_shape = pred.shape
        num_classes = original_shape[-1]
        pred = pred.reshape(-1, num_classes)
        target = target.reshape(-1)
        log_pt = F.log_softmax(pred, dim=-1)
        log_pt = log_pt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = torch.exp(log_pt)
        alpha_weight = 1.0
        if self.alpha is not None:
            if self.alpha.device != pred.device:
                self.alpha = self.alpha.to(pred.device)
            alpha_weight = self.alpha[target]
        focal_loss = -alpha_weight * ((1.0 - pt) ** self.gamma) * log_pt
        if mask is not None:
            mask = mask.reshape(-1)
            focal_loss = focal_loss * mask
        if self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


class FocalLoss_plus(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='none'):
        super(FocalLoss_plus, self).__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha, dtype=torch.float32) if alpha is not None else None
        self.reduction = reduction
    
    def forward(self, pred, target, mask=None):
        # 全部用 float32 计算 loss
        pred = pred.float()
        target = target.long()
        if mask is not None:
            mask = mask.float()

        original_shape = pred.shape
        num_classes = original_shape[-1]
        pred = pred.reshape(-1, num_classes)
        target = target.reshape(-1)
        
        log_pt = F.log_softmax(pred, dim=-1)
        pt = torch.exp(log_pt)  # [N, num_classes]
        
        # 计算 p1 (目标类别概率) 和 p2 (所有类别的最大概率)
        p1 = pt.gather(1, target.unsqueeze(1)).squeeze(1)  # [N] 目标类别概率
        log_pt = log_pt.gather(1, target.unsqueeze(1)).squeeze(1)  # [N] 目标类别log概率
        
        # 计算 p2: 所有类别的最大概率（不排除目标类别）
        p2 = pt.max(dim=-1)[0]  # [N] 所有类别的最大概率
        
        # 应用新公式: -alpha * ((1 - p1 + p2) ** gamma) * log_pt
        alpha_weight = 1.0
        if self.alpha is not None:
            if self.alpha.device != pred.device:
                self.alpha = self.alpha.to(pred.device)
            alpha_weight = self.alpha[target]
        
        focal_loss = -alpha_weight * ((1.0 - p1 + p2) ** self.gamma) * log_pt
        # focal_loss = -alpha_weight * ((0.1 - p1 + p2) ** self.gamma) * log_pt
        if mask is not None:
            mask = mask.reshape(-1)
            focal_loss = focal_loss * mask
        
        if self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss

 
def setup_optimizer_scheduler(args, model, train_loader):
    """设置优化器和学习率调度器"""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ] 
     
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.min_num)
    
    # 梯度累积下，scheduler应按“参数更新步数”计算
    t_total = len(train_loader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup( 
        optimizer,   
        num_warmup_steps=args.warmup * t_total, 
        num_training_steps=t_total
    )  
     
    return optimizer, scheduler 




 
def train_epoch(args, model, train_loader, optimizer, scheduler, epoch, unified_logger):
    """训练一个epoch"""
    model.train() 
    epoch_loss = 0
    epoch_enrl = 0
    epoch_en = 0
    epoch_htr = 0
    
 
    en_alpha = [0.1, 1.0]
    enr_alpha = [0.1, 1.0, 1.0, 1.0]
    htr_alpha = [0.1, 1.0, 1.0, 1.0, 1.0, 1.0]  
    # 初始化每类标签的总损失统计
    en_class_losses = [0.0] * 2  # EN有2个类别
    enr_class_losses = [0.0] * 4  # ENR有4个类别
    htr_class_losses = [0.0] * 6  # HTR有6个类别 

    
    total_steps = len(train_loader)
    loss_type = getattr(args, 'loss', 'FocalLoss_plus')
    gamma = 2.0

    # 根据命令行 --loss 选择损失类：CE / FocalLoss / FocalLoss_plus
    if loss_type == 'CE':
        LossCls = CrossEntropyLossWithMask
        loss_kwargs_sum = dict(alpha=None, reduction='sum')
        loss_kwargs_none = dict(alpha=None, reduction='none')
    elif loss_type == 'FocalLoss':
        LossCls = FocalLoss
        loss_kwargs_sum = dict(alpha=None, gamma=gamma, reduction='sum')
        loss_kwargs_none = dict(alpha=None, gamma=gamma, reduction='none')
    else:  # FocalLoss_plus
        LossCls = FocalLoss_plus
        loss_kwargs_sum = dict(alpha=None, gamma=gamma, reduction='sum')
        loss_kwargs_none = dict(alpha=None, gamma=gamma, reduction='none')

    # 统一构造：CE 无 gamma，FocalLoss/FocalLoss_plus 有 gamma
    def _make_loss(alpha, reduction, for_sum=True):
        kwargs = dict(alpha=alpha, reduction=reduction)
        if LossCls in (FocalLoss, FocalLoss_plus):
            kwargs['gamma'] = gamma
        return LossCls(**kwargs)
    en_loss_sum = _make_loss(en_alpha, 'sum')
    enr_loss_sum = _make_loss(enr_alpha, 'sum')
    htr_loss_sum = _make_loss(htr_alpha, 'sum')
    en_loss_none = _make_loss(en_alpha, 'none')
    enr_loss_none = _make_loss(enr_alpha, 'none')
    htr_loss_none = _make_loss(htr_alpha, 'none')

    if epoch == 0:
        unified_logger.info("=" * 80)
        unified_logger.info(f"开始训练 Epoch {epoch}")
        unified_logger.info(f"总步数: {total_steps}")
        unified_logger.info(f"损失函数: {loss_type}")
        unified_logger.info(f"ENR类别权重: {enr_alpha}")
        unified_logger.info(f"EN类别权重: {en_alpha}")
        unified_logger.info(f"HTR类别权重: {htr_alpha}")
        unified_logger.info("=" * 80)
    
    iterator = tqdm(train_loader, total=total_steps, desc=f"Epoch {epoch}")
    for step, batch in enumerate(iterator):
            # 准备数据
            optimizer.zero_grad()
            batch_data = [torch.tensor(d).to("cuda") for d in batch]
            (batch_input_ids, batch_input_mask, batch_enr_table, 
             batch_enr_mask, 
             batch_en_table, batch_en_mask, batch_htr_table, 
             batch_htr_mask) = batch_data
            del batch_data
            
            # 前向传播
            enr_table, en_table, htr_table = model(
                batch_input_ids, batch_input_mask)

            # 计算masked-sum，再用有效mask元素数归一化为"masked-mean"
            enr_loss = enr_loss_sum(enr_table, batch_enr_table.long(), batch_enr_mask.long())
            en_loss = en_loss_sum(en_table, batch_en_table.long(), batch_en_mask.long())
            htr_loss = htr_loss_sum(htr_table, batch_htr_table.long(), batch_htr_mask.long())

            total_loss = 1.0 * enr_loss +  1.0 * htr_loss + 1.0 * en_loss

            # 计算每个元素的损失（用于统计每类标签的损失）
            enr_loss_per_element = enr_loss_none(enr_table, batch_enr_table.long(), batch_enr_mask.long())
            en_loss_per_element = en_loss_none(en_table, batch_en_table.long(), batch_en_mask.long())
            htr_loss_per_element = htr_loss_none(htr_table, batch_htr_table.long(), batch_htr_mask.long())
            
            # 展平target和mask用于统计
            enr_targets = batch_enr_table.reshape(-1).long()
            en_targets = batch_en_table.reshape(-1).long()
            htr_targets = batch_htr_table.reshape(-1).long()
            enr_mask_flat = batch_enr_mask.reshape(-1).long()
            en_mask_flat = batch_en_mask.reshape(-1).long()
            htr_mask_flat = batch_htr_mask.reshape(-1).long()
            
            # 统计ENR每个类别的损失
            for class_id in range(4):
                class_mask = (enr_targets == class_id) & (enr_mask_flat > 0)
                if class_mask.any():
                    enr_class_losses[class_id] += enr_loss_per_element[class_mask].sum().item()
            
            # 统计EN每个类别的损失
            for class_id in range(2):
                class_mask = (en_targets == class_id) & (en_mask_flat > 0)
                if class_mask.any():
                    en_class_losses[class_id] += en_loss_per_element[class_mask].sum().item()
            
            # 统计HTR每个类别的损失
            for class_id in range(6):
                class_mask = (htr_targets == class_id) & (htr_mask_flat > 0)
                if class_mask.any():
                    htr_class_losses[class_id] += htr_loss_per_element[class_mask].sum().item()

            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # 优化步骤
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # 检查loss是否为NaN
            if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                unified_logger.warning(f"Epoch {epoch}, Step {step}: NaN or Inf detected in loss, skipping update")
                optimizer.zero_grad()
                continue
            
            epoch_loss += total_loss.item()
            epoch_enrl += enr_loss.item()
            epoch_en += en_loss.item()
            epoch_htr += htr_loss.item()
            iterator.set_postfix({
                'total': f'{total_loss.item():.4f}',
                'enr': f'{enr_loss:.4f}',
                'en': f'{en_loss:.4f}',
                'htr': f'{htr_loss:.4f}',
                'mem': f'{torch.cuda.memory_reserved() / 1024**3:.1f}GB'
            })
    
    avg_epoch_loss = epoch_loss / total_steps
    avg_enr_loss = epoch_enrl / total_steps
    avg_en_loss = epoch_en / total_steps
    avg_htr_loss = epoch_htr / total_steps

    unified_logger.info("=" * 80)
    unified_logger.info(f"Epoch {epoch} 训练完成")
    unified_logger.info(f"平均总损失: {avg_epoch_loss:.6f},平均ENR损失: {avg_enr_loss:.6f},平均EN损失: {avg_en_loss:.6f},平均HTR损失: {avg_htr_loss:.6f}")
    
    # 输出每类标签的总损失
    unified_logger.info("每类标签的总损失:")
    unified_logger.info(f"  ENR类别总损失: {[f'{loss:.6f}' for loss in enr_class_losses]}")
    unified_logger.info(f"  EN类别总损失: {[f'{loss:.6f}' for loss in en_class_losses]}")
    unified_logger.info(f"  HTR类别总损失: {[f'{loss:.6f}' for loss in htr_class_losses]}")
    unified_logger.info("=" * 80)
    
    return avg_epoch_loss
