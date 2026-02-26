import torch
import json
from collections import Counter
from tqdm import tqdm
from bert4keras.tokenizers import Tokenizer
from util import strip_accents 
def extract_spoes(batch, args, tokenizer, model, id2predicate, bert4tokenizer):


    _, _, batch_input_tokens, batch_ex = batch
    batch_input_ids, batch_input_mask = [torch.tensor(d).to("cuda") for d in batch[:-2]]

    model.to("cuda")
    model.eval()
    with torch.no_grad():
        enr_logits, en_logits, htr_logits = model(batch_input_ids, batch_input_mask)
        enr_logits = enr_logits.cpu().detach().numpy()
        en_logits = en_logits.cpu().detach().numpy()
        htr_logits = htr_logits.cpu().detach().numpy()
    res_id = get_pred_spo(enr_logits, en_logits, htr_logits, batch_input_ids, batch_input_tokens)
    batch_spo = [[] for _ in range(len(batch_ex))]
    for b, ex in enumerate(batch_ex):
        text = ex["text"]
        text = strip_accents(args.dataset,text)
        tokens = batch_input_tokens[b]
        mapping = bert4tokenizer.rematch(text, tokens)
        for sh, st, r, oh, ot in res_id[b]:
            if sh < len(mapping) and st < len(mapping) and oh < len(mapping) and ot < len(mapping):
                if mapping[sh] and mapping[st] and mapping[oh] and mapping[ot]:  # Check if sublists are not empty
                    s_start = mapping[sh][0]
                    s_end = mapping[st][-1]
                    o_start = mapping[oh][0]
                    o_end = mapping[ot][-1]
                    
                    if s_start <= s_end and o_start <= o_end:  # Validate indices
                        batch_spo[b].append(
                            (text[s_start:s_end + 1], id2predicate[str(r)], text[o_start:o_end + 1])
                        )

    return batch_spo

def get_pred_spo(enr_logits, en_logits, htr_logits, input_ids, input_tokens):
    B, L, R, _= enr_logits.shape
    res = []
    for i in range(B):
        res.append([])
    enr_table = enr_logits.argmax(axis=-1)

    en_table = en_logits.argmax(axis=-1)
    htr_table = htr_logits.argmax(axis=-1)      

    for b in range(B):
        for i in range(1, L):
            for j in range(1, L):
                if htr_table[b, i, j] in [3, 4, 5]:
                    for r in range(R):
                        if (enr_table[b, i, r] == 1 or enr_table[b, i, r] == 3) :
                            if (enr_table[b, j, r] == 2 or enr_table[b, j, r] == 3):
                                if en_table[b, i, i] == 1 and en_table[b, j, j] == 1:
                                    res[b].append((i, i, r, j, j))
                                            
                if htr_table[b, i, j] in [2, 5]:
                    for m in range(1, i+1):
                        for n in range(1, j+1):
                            if htr_table[b, m, n] in [1,4]:
                                for r in range(R):
                                    if (enr_table[b, i, r] == 1 or enr_table[b, i, r] == 3): 
                                        if (enr_table[b, j, r] == 2 or enr_table[b, j, r] == 3):
                                            if (enr_table[b, m, r] == 1 or enr_table[b, m, r] == 3) :
                                                if (enr_table[b, n, r] == 2 or enr_table[b, n, r] == 3):
                                                    if en_table[b, m, i] == 1 and en_table[b, n, j] == 1:
                                                        res[b].append((m, i, r, n, j))
                              

    return res





def evaluate(args, tokenizer, model, dataloader, evl_path, id2predicate, bert4tokenizer):
    # 严格三元组 (s, p, o) 完全匹配
    X, Y, Z = 1e-10, 1e-10, 1e-10
    # 实体对 (s, o) 匹配：只要求主体和客体实体对正确
    X_ep, Y_ep, Z_ep = 1e-10, 1e-10, 1e-10
    # 关系识别：只比较谓词/关系类型（按关系类型微观匹配，考虑重复）
    X_rel, Y_rel, Z_rel = 1e-10, 1e-10, 1e-10
    f = open(evl_path, 'w', encoding='utf-8')
    pbar = tqdm(dataloader, desc='eval')
    for batch in pbar:
        batch_ex = batch[-1]
        batch_spo = extract_spoes(batch, args, tokenizer, model, id2predicate, bert4tokenizer)
        for i, ex in enumerate(batch_ex):
            ex['text'] = strip_accents(args.dataset, ex['text'])
            R = set(batch_spo[i])
            T = set([(item[0], item[1], item[2]) for item in ex['triple_list']])
            R = {tuple(strip_accents(args.dataset, x) for x in t) for t in R}
            T = {tuple(strip_accents(args.dataset, x) for x in t) for t in T}
            # 严格三元组
            X += len(R & T)
            Y += len(R)
            Z += len(T)
            # 实体对：(subject, object) 集合
            R_pairs = {(t[0], t[2]) for t in R}
            T_pairs = {(t[0], t[2]) for t in T}
            X_ep += len(R_pairs & T_pairs)
            Y_ep += len(R_pairs)
            Z_ep += len(T_pairs)
            # 关系识别：按关系类型计数，每个类型最多匹配 min(预测数, 标注数)
            R_rels = Counter(t[1] for t in R)
            T_rels = Counter(t[1] for t in T)
            all_rels = set(R_rels) | set(T_rels)
            X_rel += sum(min(R_rels[r], T_rels[r]) for r in all_rels)
            Y_rel += len(R)
            Z_rel += len(T)
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            f1_ep = 2 * X_ep / (Y_ep + Z_ep)
            f1_rel = 2 * X_rel / (Y_rel + Z_rel)
            pbar.set_postfix(f1=f'{f1:.6f}', precision=f'{precision:.6f}', recall=f'{recall:.6f}')
            s = json.dumps({
                'text': ex['text'],
                'triple_list': list(T),
                'triple_list_pred': list(R),
                'new': list(R - T),
                'lack': list(T - R),
            }, ensure_ascii=False, indent=4)
            f.write(s + '\n')
    f.close()
    # 严格三元组 P/R/F1
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    # 实体对 P/R/F1
    f1_ep, precision_ep, recall_ep = 2 * X_ep / (Y_ep + Z_ep), X_ep / Y_ep, X_ep / Z_ep
    # 关系识别 P/R/F1
    f1_rel, precision_rel, recall_rel = 2 * X_rel / (Y_rel + Z_rel), X_rel / Y_rel, X_rel / Z_rel
    return f1, precision, recall, f1_ep, precision_ep, recall_ep, f1_rel, precision_rel, recall_rel

