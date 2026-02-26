import json, os
from collections import Counter

# 脚本所在目录 = CAMGT/dataset，dataset_name 为子目录名（如 WebNLG、NYT）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_name = 'WebNLG'
file_path = os.path.join(BASE_DIR, dataset_name, 'test.json')
save_dir  = os.path.join(BASE_DIR, dataset_name)
os.makedirs(save_dir, exist_ok=True)

# ---------- 读入 ----------
with open(file_path, encoding='utf-8') as f:
    sentences = json.load(f)

# ---------- 初始化 ----------
epo_list, seo_list, normal_list = [], [], []
quantity_lists = [[] for _ in range(6)]   # 0 不用，1~5 对应 n

# ---------- 遍历 ----------
for sent in sentences:
    text    = sent['text']
    triples = [list(t) for t in sent.get('triple_list', [])]  # 转回 list，方便 dump
    n       = len(triples)

    # 数量文件：n>5 的也归到 5
    bucket = min(n, 5)
    quantity_lists[bucket].append({"text": text, "triple_list": triples})

    # 判断类型
    has_epo = any(v >= 2 for v in Counter((h, t) for h,_, _, t,_ in triples).values())

    has_seo = False
    for i in range(n):
        h1,_, _, t1,_ = triples[i]
        if h1 == t1:
            has_seo = True
            break
        for j in range(i + 1, n):
            h2,_,  _, t2,_ = triples[j]
            if {h1, t1} & {h2, t2}:
                has_seo = True
                break
        if has_seo:
            break

    unique_entities = {e for h,_,  _, t,_  in triples for e in (h, t)}
    is_normal = (not has_epo and not has_seo and
                 len(unique_entities) == 2 * n and
                 not any(h in t or t in h for h,_, _, t,_ in triples))

    # 写入对应类型文件
    record = {"text": text, "triple_list": triples}
    if has_epo:
        epo_list.append(record)
    elif has_seo:
        seo_list.append(record)
    elif is_normal:
        normal_list.append(record)

# ---------- 保存 3 个类型文件（Pretty-printed） ----------
for name, lst in [('epo', epo_list), ('seo', seo_list), ('normal', normal_list)]:
    with open(os.path.join(save_dir, f'{name}.json'), 'w', encoding='utf-8') as f:
        json.dump(lst, f, ensure_ascii=False, indent=4)

# ---------- 保存 5 个数量文件（Pretty-printed） ----------
for k in range(1, 6):
    with open(os.path.join(save_dir, f'{k}.json'), 'w', encoding='utf-8') as f:
        json.dump(quantity_lists[k], f, ensure_ascii=False, indent=4)

# ---------- 打印统计 ----------
print('EPO 句子数:', len(epo_list), f'→ {os.path.join(save_dir, "epo.json")}')
print('SEO 句子数:', len(seo_list), f'→ {os.path.join(save_dir, "seo.json")}')
print('Normal 句子数:', len(normal_list), f'→ {os.path.join(save_dir, "normal.json")}')
for k in range(1, 6):
    print(f'数量 {k} 的句子数: {len(quantity_lists[k])} → {save_dir}/{k}.json')
print('句子总数:', len(sentences))