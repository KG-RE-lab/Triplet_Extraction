
import os
import json
import torch
from transformers import WEIGHTS_NAME, BertTokenizer
from bert4keras.tokenizers import Tokenizer
from util import *
from dataloader import CustomDataLoader
from evaluation_utils import evaluate
from train import *
import datetime
import os
def train(args):
    """主训练函数"""
    # set_seed()
    setup_environment(args)
    # 设置路径
    now = datetime.datetime.now()
    time_str = now.strftime("%m.%d.%H.%M")
    file_id = args.file_id
    base_output_path = os.path.join(args.base_path, args.dataset, "results", file_id)
    rel2id_path = os.path.join(args.base_path, args.dataset, "rel2id.json")
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)
    args.output_path = base_output_path
    args.dev_pred_path = os.path.join(base_output_path, "dev_pred.json")
    args.test_pred_path = os.path.join(base_output_path, "test_pred.json")
    args.train_pred_path = os.path.join(base_output_path, "train_pred.json")  
    args.log_path = os.path.join(base_output_path, "log.txt")
    
    # 初始化统一日志系统（所有日志写入log.txt，覆盖模式）
    unified_logger = UnifiedLogger(log_file=args.log_path)
    # 只在控制台输出日志文件路径（仅一次）
    print(f"日志文件路径: {args.log_path}")
    unified_logger.info("=" * 80)
    unified_logger.info("训练开始")
    unified_logger.info(f"输出路径: {base_output_path}")
    unified_logger.info(f"日志文件: {args.log_path}")
    unified_logger.info(f"损失函数: {getattr(args, 'loss', 'FocalLoss_plus')}")
    # 消融实验设置：不传 --ablate 为完整模型；传了则记录已关闭的模块
    ablate_list = getattr(args, 'ablate', []) or []
    if ablate_list:
        unified_logger.info(f"消融实验: 已关闭模块 {ablate_list}")
    else:
        unified_logger.info("消融实验: 无（完整模型，不消除任何模块）")
    unified_logger.info("=" * 80)
    
    # 加载数据
    id2predicate, predicate2id = json.load(open(rel2id_path))
    bert4tokenizer = Tokenizer(args.bert_vocab_path)
    tokenizer = BertTokenizer(vocab_file=args.bert_vocab_path, do_lower_case=False)
    # 加载模型
    model = load_config_and_model(args, id2predicate)
    # 准备数据加载器
    dataloader = CustomDataLoader(args)
    train_loader = dataloader.get_dataloader(data_sign='train')
    dev_loader = dataloader.get_dataloader(data_sign='dev')
    test_loader = dataloader.get_dataloader(data_sign='test')
    optimizer, scheduler = setup_optimizer_scheduler(args, model, train_loader)
    print_config(args,file_id)
    best_f1 = -1.0

    for epoch in range (args.num_train_epochs):
        # 训练一个epoch
        avg_loss = train_epoch(args, model, train_loader, optimizer, scheduler, epoch, unified_logger)
  
        # 在验证集上评估（严格三元组 + 实体对 + 关系识别）
        f1, precision, recall, f1_ep, precision_ep, recall_ep, f1_rel, precision_rel, recall_rel = evaluate(
            args, tokenizer, model, dev_loader, 
            args.dev_pred_path, id2predicate, bert4tokenizer 
        )
        print(f"日志文件路径: {args.log_path}")
        # 保存最佳模型
        if f1 > best_f1: 
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(args.output_path, WEIGHTS_NAME))
            unified_logger.info(f"Epoch {epoch}: 保存最佳模型 (Dev F1: {f1:.6f})")
        # 记录epoch评估结果（包含训练性能指标）
        unified_logger.info(f"Epoch {epoch} 评估结果:")
        unified_logger.info(f"  训练损失: {avg_loss:.6f}")
        unified_logger.info(f"  Dev 严格三元组 F1: {f1:.6f}, P: {precision:.6f}, R: {recall:.6f}")
        unified_logger.info(f"  最佳F1: {best_f1:.6f}")
        unified_logger.info("-" * 80)
    # 最终测试  
    unified_logger.info("=" * 80)
    unified_logger.info("开始最终测试")
    model.load_state_dict(torch.load(
        os.path.join(args.output_path, WEIGHTS_NAME), 
        map_location="cuda"
    ))

    f1, precision, recall, f1_ep, precision_ep, recall_ep, f1_rel, precision_rel, recall_rel = evaluate(
        args, tokenizer, model, test_loader, 
        args.test_pred_path, id2predicate, bert4tokenizer
    )
    unified_logger.info("=" * 80)
    unified_logger.info(f"最终测试结果:")
    unified_logger.info(f"  严格三元组 - F1: {f1:.6f}, P: {precision:.6f}, R: {recall:.6f}")
    unified_logger.info(f"  实体对     - F1: {f1_ep:.6f}, P: {precision_ep:.6f}, R: {recall_ep:.6f}")
    unified_logger.info(f"  关系识别   - F1: {f1_rel:.6f}, P: {precision_rel:.6f}, R: {recall_rel:.6f}")
    unified_logger.info("=" * 80)
    unified_logger.info("训练完成")
    unified_logger.info("=" * 80)
def test(args):
    """测试函数"""
    setup_environment(args)
    # 设置路径  
    output_path = os.path.join(args.base_path, args.dataset, "results", args.file_id)
    rel2id_path = os.path.join(args.base_path, args.dataset, "rel2id.json")
    result_file_path = os.path.join(output_path, "out.txt")  # 结果保存文件
    # 加载数据
    id2predicate, predicate2id = json.load(open(rel2id_path))
    bert4tokenizer = Tokenizer(args.bert_vocab_path)
    tokenizer = BertTokenizer(vocab_file=args.bert_vocab_path, do_lower_case=False)
    # 加载模型
    model = load_config_and_model(args, id2predicate)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    weight_file = os.path.join(output_path, WEIGHTS_NAME)
    if os.path.isfile(weight_file):
        model.load_state_dict(torch.load(weight_file, map_location="cuda"), strict=False)
        
    print_config(args, args.file_id)
    dataloader = CustomDataLoader(args)
    # 测试多个数据集 
    # test_signs = ["test"]
    test_signs = ["test", "1", "2", "3", "4", "5", "epo", "normal", "seo"]
    results = {}
    for sign in test_signs:
        print(f"\n========== Evaluating {sign.upper()} ==========")
        test_pred_path = os.path.join(output_path, f"{sign}.json")
        test_loader = dataloader.get_dataloader(data_sign=sign)
        # 加载权重
        model.load_state_dict(torch.load(weight_file, map_location="cuda"), strict=False)
        # 评估（严格三元组 + 实体对 + 关系识别）
        f1, precision, recall, f1_ep, precision_ep, recall_ep, f1_rel, precision_rel, recall_rel = evaluate(
            args, tokenizer, model, test_loader,
            test_pred_path, id2predicate, bert4tokenizer,
        )
        results[sign] = (f1, precision, recall, f1_ep, precision_ep, recall_ep, f1_rel, precision_rel, recall_rel)
        print(f"[{sign}] 严格三元组 f1:{f1:.6f}, P:{precision:.6f}, R:{recall:.6f}")
        print(f"[{sign}] 实体对     f1:{f1_ep:.6f}, P:{precision_ep:.6f}, R:{recall_ep:.6f}")
        print(f"[{sign}] 关系识别   f1:{f1_rel:.6f}, P:{precision_rel:.6f}, R:{recall_rel:.6f}")
    # 输出汇总结果到控制台
    print("\n========== Summary (严格三元组) ==========")
    for sign, (f1, p, r, *_ ) in results.items():
        print(f"{sign:>6}: f1={f1:.4f}, P={p:.4f}, R={r:.4f}")
    print("\n========== Summary (实体对) ==========")
    for sign, (_, _, _, f1_ep, p_ep, r_ep, *_) in results.items():
        print(f"{sign:>6}: f1={f1_ep:.4f}, P={p_ep:.4f}, R={r_ep:.4f}")
    print("\n========== Summary (关系识别) ==========")
    for sign, (_, _, _, _, _, _, f1_rel, p_rel, r_rel) in results.items():
        print(f"{sign:>6}: f1={f1_rel:.4f}, P={p_rel:.4f}, R={r_rel:.4f}")
    # 保存结果到文件
    with open(result_file_path, "w", encoding="utf-8") as f:
        f.write("========== Evaluation Results ==========\n")
        f.write(f"Model: {args.file_id}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write("=" * 50 + "\n\n")
        f.write("严格三元组 (Strict Triple):\n")
        f.write("-" * 50 + "\n")
        for sign, (f1, p, r, f1_ep, p_ep, r_ep, f1_rel, p_rel, r_rel) in results.items():
            f.write(f"{sign:>6}: F1={f1:.6f}, P={p:.6f}, R={r:.6f}\n")
        f.write("\n实体对 (Entity Pair):\n")
        f.write("-" * 50 + "\n")
        for sign, (f1, p, r, f1_ep, p_ep, r_ep, f1_rel, p_rel, r_rel) in results.items():
            f.write(f"{sign:>6}: F1={f1_ep:.6f}, P={p_ep:.6f}, R={r_ep:.6f}\n")
        f.write("\n关系识别 (Relation):\n")
        f.write("-" * 50 + "\n")
        for sign, (f1, p, r, f1_ep, p_ep, r_ep, f1_rel, p_rel, r_rel) in results.items():
            f.write(f"{sign:>6}: F1={f1_rel:.6f}, P={p_rel:.6f}, R={r_rel:.6f}\n")
        f.write("\n" + "=" * 50 + "\n")
    print(f"\nResults saved to: {result_file_path}")