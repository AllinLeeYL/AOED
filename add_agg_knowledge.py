import json, sys, jsonlines, re
# import nltk
from functions import get_bert
from sqlova.utils.utils_wikisql import load_wikisql_data

# nltk.download('punkt')

prog = re.compile('##.*')
prog_est = re.compile('(.*est$)')
prog_hodr = re.compile('(.*est$)|(least)|(most)')

agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
"""
agg_texts = [[],\
             ['maximum', 'the largest', 'the highest', 'the most'],\
             ['the earliest', 'the lowest', 'the least', 'the smallest'],\
             ['games', 'times'],\
             ['sum'],\
             ['average']]
agg_headers = [[],\
               ['max', 'highest', 'largest', 'most'],\
               ['min', 'lowest', 'least', 'earliest'],\
               ['count', 'games', 'times'],\
               ['sum'],\
               ['avg', 'average']]
"""
# Here we only use tokenizer from bert model
model_bert, tokenizer, bert_config = get_bert('./data_and_model', 'uncased_L-12_H-768_A-12', True, False)
"""
for i, agg_text in enumerate(agg_texts):
    for j in range(0, len(agg_text)):
        agg_texts[i][j] = tokenizer.tokenize(agg_text[j])
for i, agg_header in enumerate(agg_headers):
    for j in range(0, len(agg_header)):
        agg_headers[i][j] = tokenizer.tokenize(agg_header[j])
"""

def containText(text, toks):
    # if a str in question_toks
    # return [-1] for failure, index of text in question_toks for success
    for i in range(0, len(toks)):
        flag_t = 1
        for j in range(0, len(text)):
            if i + j >= len(toks) or toks[i + j] != text[j]:
                flag_t = 0
                break
        if flag_t == 1 and i + len(text) < len(toks) and prog.match(toks[i + len(text)]) != None:
            flag_t = 0
            break
        if flag_t == 1:
            return [i + j for j in range(0, len(text))]
    return [-1]

def _addAggKnowledgeForOne(one_data, table, agg_idx, question_toks, agg_hds, agg_texts):
    # return (status, idx)
    # status: 0 -- success | 1 -- not added | 2 -- error
    # idx: index of text related to a AGG
    table_headers = table['header']
    # table_types = table['types']
    # table_rows = table['rows']
    # sel = one_data['sql']['sel']

    if len(question_toks) != len(one_data['bertindex_knowledge']):
        # len(knowledge) not equal to len(tokens), just pass through it
        return 2, -1 # error
    else:
        one_agg_texts = agg_texts[agg_idx] # one AGG text to a AGG OP
        one_agg_headers = agg_hds[agg_idx] # one AGG text to a AGG OP
        # if table headers contain the token, then not add it
        for one_agg_header in one_agg_headers:
            tbhtoks = []
            for table_header in table_headers:
                table_header_toks = tokenizer.tokenize(table_header)
                tbhtoks.append(table_header_toks)
                if containText(one_agg_header, table_header_toks)[0] != -1:
                    return 1, -1 # not added
        
        # add 'est'
        for idx_t, tok in enumerate(question_toks):
            if prog_est.match(tok) != None:
                one_data['bertindex_knowledge'][idx_t] = 6
                # return 0, 0
        

        for i, one_agg_text in enumerate(one_agg_texts):
            idx = containText(one_agg_text, question_toks) # check if question contains special tokens
            if idx[0] != -1:
                # if one_data['sql']['agg'] != agg_idx and (agg_idx == 3 or agg_idx == 4):
                    # print(tbhtoks)
                    # print(one_data['question'], question_toks, table_headers, agg_ops[agg_idx], agg_ops[one_data['sql']['agg']], end='\n\n')
                    # print(question_toks, table_headers[sel], table_types[sel], agg_ops[agg_idx], agg_ops[one_data['sql']['agg']], '\n', [row[sel] for row in table_rows], end='\n\n')
                for j in range(0, len(idx)):
                    one_data['bertindex_knowledge'][idx[j]] = 5
                return 0, i # success
    return 1, -1 # not added

def addAggKnowledge(data, tables, mode, agg_hds, agg_texts):
    # Here we only use tokenizer from bert model
    model_bert, tokenizer, bert_config = get_bert('./data_and_model', 'uncased_L-12_H-768_A-12', True, False)
    for i in range(0, len(data)):
        # Tokenize
        # Because the tokens in 'train_knowledge.jsonl' is diffferent from the output of bert tokenizer
        question_toks = tokenizer.tokenize(data[i]['question'])
        table = tables[data[i]['table_id']]
        # add AGG knowledge
        if mode == 'dev' or mode == 'train' or mode =='test':
            for agg_idx in range(1, len(agg_ops)):
                status, text_idx = _addAggKnowledgeForOne(data[i], table, agg_idx, question_toks, agg_hds, agg_texts)
        else:
            print('wrong!')
            return data
    return data
            
def print_res(global_stcs):
    for i in range(0, len(agg_ops)):
        print(agg_ops[i], '\t-- success =', sum(global_stcs[i][0]), global_stcs[i][0], 
                          '\tnot_added =', global_stcs[i][1], 
                          '\terror =', global_stcs[i][2], 
                          '\ttotal =', sum(global_stcs[i][0]) + global_stcs[i][1] + global_stcs[i][2])
    print('Global\t-- success =', sum(sum(global_stcs[i][0]) for i in range(0, len(agg_ops))), 
          '\tnot_added =', sum(global_stcs[i][1] for i in range(0, len(agg_ops))), 
          '\terror =', sum(global_stcs[i][2] for i in range(0, len(agg_ops)))
         )

def gather_info():
    mode = 'train'
    if len(sys.argv) == 2:
        mode = sys.argv[1]
    
    agg_NLQ_num = [0 for agg_op in agg_ops]
    agg_toks_raw = [{} for agg_op in agg_ops]
    agg_toks = [[] for agg_op in agg_ops]
    data, tables = load_wikisql_data('./data_and_model', mode=mode, toy_model=False, toy_size=12, no_hs_tok=True)

    if mode == 'train':
        for idx, one_data in enumerate(data):
            agg_idx = one_data['sql']['agg']
            q_toks = tokenizer.tokenize(one_data['question'])
            for q_tok in q_toks:
                if q_tok not in agg_toks_raw[agg_idx].keys():
                    agg_toks_raw[agg_idx][q_tok] = 1
                else:
                    agg_toks_raw[agg_idx][q_tok] += 1
            agg_NLQ_num[agg_idx] += 1
    
        # use only 10% mostly-used words
        for i, agg_tok in enumerate(agg_toks_raw):
            agg_tok = [[tok, agg_tok[tok]] for tok in agg_tok.keys()]
            agg_tok = sorted(agg_tok, key=lambda tok : tok[1], reverse=True)
            agg_toks[i] = agg_tok[0: int(len(agg_tok) / 90)]
        # delete crossed words
        for i, agg_tok in enumerate(agg_toks):
            agg_toks[i] = set([tok_pair[0] for tok_pair in agg_tok])
        noise = set()
        for i in range(0, len(agg_toks)):
            for j in range(i + 1, len(agg_toks) - 1):
                noise = noise | (agg_toks[i] & agg_toks[j])
        # remove noise
        for i, agg_tok in enumerate(agg_toks):
            agg_toks[i] = list(agg_tok - noise)
            for j, tok in enumerate(agg_toks[i]):
                agg_toks[i][j] = [tok, agg_toks_raw[i][tok], 0, 0] # [str, num, num, num]
        all_tok_freq = [sum([agg_tok[key] for key in agg_tok.keys()]) for agg_tok in agg_toks_raw]
        for idx, one_data in enumerate(data):
            q_toks = tokenizer.tokenize(one_data['question'])
            for i in range(0, len(agg_toks)):
                # every agg
                for j, tok in enumerate(agg_toks[i]):
                    # every tok related to one agg. tok -- [str, num, num, num]
                    if tok[0] in q_toks:
                        if one_data['sql']['agg'] == i:
                            agg_toks[i][j][2] += 1
                        else:
                            agg_toks[i][j][3] += 1
        with open('data_and_model/agg_rl_toks.json', mode='w') as f:
            json.dump([agg_toks, all_tok_freq, agg_NLQ_num], f)
    else:
        print('omitted, only execute gather_info when mode == train')
    

def add_knlg():
    mode = 'train'
    if len(sys.argv) == 2:
        mode = sys.argv[1]

    # load dataset and gathered info
    data, tables = load_wikisql_data('./data_and_model', mode=mode, toy_model=False, toy_size=12, no_hs_tok=True)
    with open('data_and_model/agg_rl_toks.json', mode='r') as f:
        [agg_toks, all_tok_freq, agg_NLQ_num] = json.load(f)
    # preprocess
    agg_texts = [[] for t in agg_ops]
    agg_hds = [[] for t in agg_ops]
    for i, agg_tok in enumerate(agg_toks):
        if i == 0:
            agg_toks[i] = []
        else:
            toks_texts = [tok[0] for tok in agg_tok]
            toks_hds = [tok[0] for tok in agg_tok]
            for j, tok in enumerate(toks_texts):
                if prog_hodr.match(tok) != None:
                    tok_text = 'the ' + tok
                else:
                    tok_text = tok
                toks_texts[j] = tokenizer.tokenize(tok_text)
                toks_hds[j] = tokenizer.tokenize(tok)
            agg_texts[i] = toks_texts
            agg_hds[i] = toks_hds + [tokenizer.tokenize(agg_ops[i])]
    # generate AGG enhanced knowledge
    data = addAggKnowledge(data, tables, mode, agg_hds, agg_texts)
    # save enhanced knowledge
    with jsonlines.open('./data_and_model/' + mode + '_knowledge_agg_enhanced.jsonl', mode='w') as writer:
        writer.write_all(data)

if __name__ == "__main__":
    gather_info()
    print('\033[1;31m gather info finished! \033[0m')
    add_knlg()
    print('\033[1;31m add enhanced knowledge finished! \033[0m')
