import encodings
import enum
import json, os, sys, jsonlines, re
from functions import get_bert
from sqlova.utils.utils_wikisql import load_wikisql_data

prog = re.compile('##.*')
prog_est = re.compile('.*est$')

agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
"""
agg_texts = [[],\
             ['maximum'],\
             ['minimum', 'the lowest', 'the least'],\
             ['count', 'number of', 'how many'],\
             ['total', 'how many'],\
             ['average']]
agg_headers = [[],\
               ['maximum', 'max'],\
               ['minimum', 'min', 'lowest', 'least'],\
               ['count'],\
               ['sum'],\
               ['average', 'avg']]
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
# Here we only use tokenizer from bert model
model_bert, tokenizer, bert_config = get_bert('./data_and_model', 'uncased_L-12_H-768_A-12', True, False)
for i, agg_text in enumerate(agg_texts):
    for j in range(0, len(agg_text)):
        agg_texts[i][j] = tokenizer.tokenize(agg_text[j])
for i, agg_header in enumerate(agg_headers):
    for j in range(0, len(agg_header)):
        agg_headers[i][j] = tokenizer.tokenize(agg_header[j])

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

def _addAggKnowledgeForOne(one_data, table, agg_idx, question_toks):
    # return (status, idx)
    # status: 0 -- success | 1 -- not added | 2 -- error
    # idx: index of text related to a AGG
    table_headers = table['header']
    table_types = table['types']
    table_rows = table['rows']
    sel = one_data['sql']['sel']

    if len(question_toks) != len(one_data['bertindex_knowledge']):
        # len(knowledge) not equal to len(tokens), just pass through it
        return 2, -1 # error
    else:
        one_agg_texts = agg_texts[agg_idx] # one AGG text to a AGG OP
        one_agg_headers = agg_headers[agg_idx] # one AGG text to a AGG OP
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

def addAggKnowledge(data, tables, mode, global_stcs):
    # Here we only use tokenizer from bert model
    model_bert, tokenizer, bert_config = get_bert('./data_and_model', 'uncased_L-12_H-768_A-12', True, False)
    for i in range(0, len(data)):
        # Tokenize
        # Because the tokens in 'train_knowledge.jsonl' is diffferent from the output of bert tokenizer
        question_toks = tokenizer.tokenize(data[i]['question'])
        table = tables[data[i]['table_id']]
        # add AGG knowledge
        if mode == '...':
            agg_idx = data[i]['sql']['agg']
            if agg_idx != 0:
                status, text_idx = _addAggKnowledgeForOne(data[i], table, agg_idx=agg_idx, question_toks=question_toks)
                # gather statistics info
                if status == 0: # success
                    global_stcs[agg_idx][status][text_idx] += 1
                else:
                    global_stcs[agg_idx][status] += 1
            else:
                global_stcs[0][1] += 1
        elif mode == 'dev' or mode == 'train':
            for agg_idx in range(1, len(agg_ops)):
                status, text_idx = _addAggKnowledgeForOne(data[i], table, agg_idx=agg_idx, question_toks=question_toks)
                # gather statistics info
                if status == 0: # success
                    global_stcs[agg_idx][status][text_idx] += 1
                    # break
                else:
                    global_stcs[agg_idx][status] += 1
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
    

def add_knlg():
    mode = 'train'
    if len(sys.argv) == 2:
        mode = sys.argv[1]
    # load data and tables
    data, tables = load_wikisql_data('./data_and_model', mode=mode, toy_model=False, toy_size=12, no_hs_tok=True)
    # statistics: num_success[agg_texts], num_not_added, num_error
    global_stcs = [[[0 for j in range(0, len(agg_texts[i]))], 0, 0] for i in range(0, len(agg_ops))] 
    # generate AGG enhanced knowledge
    data = addAggKnowledge(data, tables, mode, global_stcs)
    # print results
    print_res(global_stcs)
    # save enhanced knowledge
    with jsonlines.open('./data_and_model/' + mode + '_knowledge_agg_enhanced.jsonl', mode='w') as writer:
        writer.write_all(data)
    # gather_info()

def add_knlg_v2():
    mode = 'train'
    if len(sys.argv) == 2:
        mode = sys.argv[1]
    
    with open('data_and_model/agg_rl_toks.json', mode='r') as f:
        agg_rl_toks = json.load(f)[1:]
    data, tables = load_wikisql_data('./data_and_model', mode=mode, toy_model=False, toy_size=12, no_hs_tok=True)

    for data_idx, one_data in enumerate(data):
        q_toks = tokenizer.tokenize(one_data['question'])
        table = tables[one_data['table_id']]
        if len(one_data['bertindex_knowledge']) != len(q_toks):
            continue
        for i, q_tok in enumerate(q_toks):
            for j, agg_rl_tok in enumerate(agg_rl_toks):
                if q_tok in agg_rl_tok:
                    one_data['bertindex_knowledge'][i] = 5
    with jsonlines.open('./data_and_model/' + mode + '_knowledge_agg_enhanced.jsonl', mode='w') as writer:
        writer.write_all(data)
    print(agg_rl_toks)

if __name__ == "__main__":
    # add_knlg()
    gather_info()
    # print('gather info finished!')
    # add_knlg_v2()
    # print('add enhanced knowledge finished!')
