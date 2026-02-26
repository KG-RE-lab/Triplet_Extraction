import json
from multiprocessing import Pool
import functools
from util import *


class InputExample(object):
    def __init__(self, text, en_pair_list, re_list, ex):
        self.text = text
        self.en_pair_list = en_pair_list
        self.re_list = re_list
        self.ex = ex


class InputFeatures(object):
    def __init__(self,
                 input_ids,
                 input_mask,
                 input_tokens=None,
                 enr_table=None,
                 enr_mask=None,              
                 en_table=None,
                 en_mask=None,
                 htr_table=None,
                 htr_mask=None,

                 ex=None,
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_tokens = input_tokens
        self.enr_table = enr_table
        self.enr_mask = enr_mask        
        self.en_table = en_table
        self.en_mask = en_mask
        self.htr_table = htr_table
        self.htr_mask = htr_mask
        self.ex = ex

def json2list(data_dir, data_sign):
    examples = []
    data_path = os.path.join(data_dir, data_sign + ".json")
    with open(data_path, "r", encoding='utf-8') as f:
        data = json.load(f)
        for sample in data:
            text = sample['text']
            en_pair_list = []
            re_list = []
            for triple in sample['triple_list']:
                en_pair_list.append([triple[0], triple[-1]])
                re_list.append(triple[1])
            example = InputExample(text=text, en_pair_list=en_pair_list, re_list=re_list, ex=sample)


            examples.append(example)
    return examples


def read_examples(data, args, tokenizer, data_sign, rel2idx):
    max_len = args.max_len
    data.text = strip_accents(args.dataset, data.text)
    input_tokens = tokenizer.tokenize(data.text)
    input_ids = tokenizer.tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_tokens)
    if len(input_tokens) > max_len:
        input_tokens = input_tokens[:max_len]
        input_ids = input_ids[:max_len]
        input_mask = input_mask[:max_len]
    n = len(input_tokens)
    if type(input_ids) is np.ndarray:
        input_ids = input_ids.tolist()

    if data_sign == 'train':
        enr_table = np.zeros([n, len(rel2idx)])
        enr_mask = np.ones((n, len(rel2idx)))
 
        en_table = np.zeros([n, n])
        en_mask = np.ones((n, n))
        # 掩码下三角部分（i > j 的位置，不包括对角线）
        # 对角线（i == j）保留为有用信息，不被掩码
        en_mask[np.tril_indices(n, k=-1)] = 0
        htr_table = np.zeros([n, n])
        htr_mask = np.ones((n, n))
        
        # Process entity-relation pairs
        for en_pair, rel in zip(data.en_pair_list, data.re_list):
            s, o = en_pair[0], en_pair[-1]
            s = strip_accents(args.dataset, s)
            o = strip_accents(args.dataset, o)
            p = rel2idx[rel]
            s = tokenizer.encode(s)[0][1:-1]
            p = rel2idx[rel]
            o = tokenizer.encode(o)[0][1:-1]
            # print("s1: ", s, " o1: ", o, " p1: ", p)
            s_idx = search(s, input_ids)
            o_idx = search(o, input_ids)
            s = (s_idx, s_idx + len(s) - 1)
            o = (o_idx, o_idx + len(o) - 1)
            s_h, s_t = s
            o_h, o_t = o
            
            if s_h == -1 or o_h == -1:
                continue
 
            if enr_table[s_h, p] == 0:
                enr_table[s_h, p] = 1
            elif enr_table[s_h, p] == 2:
                enr_table[s_h, p] = 3
            
            if s_h !=s_t:
                if enr_table[s_t, p] == 0:
                    enr_table[s_t, p] = 1
                elif enr_table[s_t, p] == 2:
                    enr_table[s_t, p] = 3

            if enr_table[o_h, p] == 0: 
                enr_table[o_h, p] = 2
            elif enr_table[o_h, p] == 1:
                enr_table[o_h, p] = 3

            if o_h != o_t:
                if enr_table[o_t, p] == 0:
                    enr_table[o_t, p] = 2
                elif enr_table[o_t, p] == 1:
                    enr_table[o_t, p] = 3
        

                      


                
            en_table[s_h, s_t] = 1
            en_table[o_h, o_t] = 1 

            if s_h == s_t and o_h == o_t:
                if htr_table[s_h, o_h] == 0:
                    htr_table[s_h, o_h] = 3
                elif htr_table[s_h, o_h] == 1:
                    htr_table[s_h, o_h] = 4
                elif htr_table[s_h, o_h] == 2:
                    htr_table[s_h, o_h] = 5  
            else:
                if htr_table[s_h, o_h] == 0:
                    htr_table[s_h, o_h] = 1
                elif htr_table[s_h, o_h] == 3:
                    htr_table[s_h, o_h] = 4
                if htr_table[s_t, o_t] == 0:
                    htr_table[s_t, o_t] = 2
                elif htr_table[s_t, o_t] == 3:
                    htr_table[s_t, o_t] = 5



        return InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            enr_table=enr_table,
            enr_mask=enr_mask,           
            en_table=en_table,
            en_mask=en_mask,
            htr_table=htr_table,
            htr_mask=htr_mask,

        )
    else:
        return InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            input_tokens=input_tokens,
            ex=data.ex
        )


def mul_process(args, tokenizer, data_dir, data_sign, rel2idx):
    data = json2list(data_dir, data_sign)

    with Pool(10) as p:
        convert_func = functools.partial(read_examples, args=args, tokenizer=tokenizer, data_sign=data_sign,
                                         rel2idx=rel2idx)
        features = p.map(func=convert_func, iterable=data)
    return features
