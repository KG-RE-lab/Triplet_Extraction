import argparse
from main import *

parser = argparse.ArgumentParser(description='Model Controller')
parser.add_argument('--cuda_id', default="3", type=str)
parser.add_argument('--train', default="train", type=str)
parser.add_argument('--train_batch_size', default=6, type=int)
parser.add_argument('--val_batch_size', default=6, type=int)
parser.add_argument('--test_batch_size', default=6, type=int)
parser.add_argument('--learning_rate', default=2e-5, type=float)
parser.add_argument('--num_train_epochs', default=100, type=int)
parser.add_argument('--fix_bert_embeddings', default=False, type=bool)
parser.add_argument('--bert_vocab_path', default="../bert-large-cased/vocab.txt", type=str)
parser.add_argument('--bert_config_path', default="../bert-large-cased/config.json", type=str)
parser.add_argument('--bert_model_path', default="../bert-large-cased/pytorch_model.bin", type=str)
parser.add_argument('--max_len', default=100, type=int)
parser.add_argument('--warmup', default=0.0, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--max_grad_norm', default=1.0, type=float)
parser.add_argument('--min_num', default=1e-7, type=float)
parser.add_argument('--base_path', default="../dataset", type=str)
parser.add_argument('--dataset', default='WebNLG', type=str)
parser.add_argument('--file_id', default="WebNLG4", type=str)   

# 损失函数选择：CE（交叉熵）、FocalLoss_plus、FocalLoss（标准 Focal Loss）
parser.add_argument('--loss', default='FocalLoss_plus', type=str,
                    choices=['CE', 'FocalLoss_plus', 'FocalLoss'],
                    help='损失函数: CE=交叉熵, FocalLoss_plus=改进 Focal Loss, FocalLoss=标准 Focal Loss')

# 消融实验：不传 --ablate 即为不消融（完整模型）；传了则关闭对应模块，可多选
# 可选值: cross_attention, reasoning, table_enhance
parser.add_argument('--ablate', nargs='*', default=[], metavar='MODULE',
                    choices=['cross_attention', 'reasoning', 'table_enhance'],
                    help='消融实验。不传则不消融（完整模型）。传则关闭模块，可多选：cross_attention, reasoning, table_enhance。例：--ablate cross_attention；--ablate cross_attention reasoning')

# python run.py --dataset=WebNLG  --cuda_id=5   --train=train  --file_id
# pkill -f "run.py.*NYT.*cuda_id=5"
args = parser.parse_args()  
 
if args.train == "train": 
    train(args)   
else:
    test(args)
