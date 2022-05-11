from turtle import left
from numpy import argmax
import json, sys
import matplotlib.pyplot as plt

agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']

def load_results():
    res_path = 'res/results_for_plot.json'
    if len(sys.argv) == 2:
        res_path = sys.argv[1]
    with open(res_path, 'r') as f:
        results = json.load(f)
    return results

def plot_results(results):
    plt.figure(figsize=(8, 6), dpi=144)
    # LOSS
    plt.title('Loss')
    plt.ylabel('average loss')
    plt.xlabel('epoch')
    ave_loss_train = [results[i]['train']['ave_loss'] for i in range(0, len(results))]
    ave_loss_dev = [results[i]['dev']['ave_loss'] for i in range(0, len(results))]
    plt.plot([i+1 for i in range(len(ave_loss_train))], ave_loss_train, '--', label='train')
    plt.plot([i+1 for i in range(len(ave_loss_dev))], ave_loss_dev, '--', label='dev')
    plt.legend()
    plt.savefig('fig/ave_loss.png')

    # GLOBAL ACC
    plt.figure(figsize=(8, 6), dpi=144)
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    acc_lx_train = [results[i]['train']['acc_lx'] for i in range(0, len(results))]
    acc_x_train = [results[i]['train']['acc_x'] for i in range(0, len(results))]
    acc_lx_dev = [results[i]['dev']['acc_lx'] for i in range(0, len(results))]
    acc_x_dev = [results[i]['dev']['acc_x'] for i in range(0, len(results))]
    plt.plot([i+1 for i in range(len(acc_lx_train))], acc_lx_train, '--', label='logic form (train)')
    plt.plot([i+1 for i in range(len(acc_x_train))], acc_x_train, '--', label='execution (train)')
    plt.plot([i+1 for i in range(len(acc_lx_dev))], acc_lx_dev, '--', label='logic form (dev)')
    plt.plot([i+1 for i in range(len(acc_x_dev))], acc_x_dev, '--', label='execution (dev)')
    lx_idx = argmax(acc_lx_dev)
    x_idx = argmax(acc_x_dev)
    plt.annotate(xy=(lx_idx, max(acc_lx_dev)),
                 text='epoch:'+str(lx_idx + 1)+'\nacc_lx:'+str(round(acc_lx_dev[lx_idx], 3)) )
    plt.annotate(xy=(x_idx, max(acc_x_dev)), 
                 text='epoch:'+str(x_idx + 1)+'\nacc_x:'+str(round(acc_x_dev[x_idx], 3)) )
    plt.legend()
    plt.savefig('fig/acc.png')

    # DEV ACC DETIAL
    plt.figure(figsize=(8, 6), dpi=144)
    plt.title('Accuracy Details')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    acc_sc_dev = [results[i]['dev']['acc_sc'] for i in range(0, len(results))]
    acc_sa_dev = [results[i]['dev']['acc_sa'] for i in range(0, len(results))]
    acc_wn_dev = [results[i]['dev']['acc_wn'] for i in range(0, len(results))]
    acc_wc_dev = [results[i]['dev']['acc_wc'] for i in range(0, len(results))]
    acc_wo_dev = [results[i]['dev']['acc_wo'] for i in range(0, len(results))]
    acc_wv_dev = [results[i]['dev']['acc_wv'] for i in range(0, len(results))]
    plt.plot([i+1 for i in range(len(acc_sc_dev))], acc_sc_dev, '--', label='SEL COLUMN')
    plt.plot([i+1 for i in range(len(acc_sa_dev))], acc_sa_dev, '--', label='SEL AGG')
    plt.plot([i+1 for i in range(len(acc_wn_dev))], acc_wn_dev, '--', label='WHERE NUM')
    plt.plot([i+1 for i in range(len(acc_wc_dev))], acc_wc_dev, '--', label='WHERE COLUMN')
    plt.plot([i+1 for i in range(len(acc_wo_dev))], acc_wo_dev, '--', label='WHERE OPERATOR')
    plt.plot([i+1 for i in range(len(acc_wv_dev))], acc_wv_dev, '--', label='WHERE VALUE')
    plt.annotate(xy=(argmax(acc_sa_dev), max(acc_sa_dev)),
                 text='epoch:'+str(argmax(acc_sa_dev) + 1)+'\nacc_sa:'+str(round(max(acc_sa_dev), 3)) )
    plt.legend()
    plt.savefig('fig/acc_detial.png')
    # SAVE
    return lx_idx, x_idx

def print_best(results, lx_idx, x_idx):
    print(results[lx_idx]['dev'])
    print(results[x_idx]['dev'])

def plot_one_res():
    results = load_results()
    lx_idx, x_idx = plot_results(results)
    print_best(results, lx_idx, x_idx)

def plot_acc_detail_compare():
    with open(sys.argv[1], 'r') as f:
        res_NL2SQL = json.load(f)
    with open(sys.argv[2], 'r') as f:
        res_ATEP = json.load(f)
    # prepare
    acc_lx_NL2SQL = [res_NL2SQL[i]['dev']['acc_lx'] for i in range(0, len(res_NL2SQL))]
    acc_lx_ATEP = [res_ATEP[i]['dev']['acc_lx'] for i in range(0, len(res_ATEP))]
    idx_NL2SQL = argmax(acc_lx_NL2SQL) # Index of the best epoch
    idx_ATEP = argmax(acc_lx_ATEP) # Index of the best epoch
    acc_detail_NL2SQL = res_NL2SQL[idx_NL2SQL]['dev']
    acc_detail_ATEP = res_ATEP[idx_ATEP]['dev']
    # plot values
    acc_names = ['acc_sc', 'acc_sa', 'acc_wn', 'acc_wc', 'acc_wo', 'acc_wv']
    acc_values_NL2SQL = [acc_detail_NL2SQL[name]*100 for name in acc_names]
    acc_values_ATEP = [acc_detail_ATEP[name]*100 for name in acc_names]    
    # plot
    plt.figure(figsize=(8, 6), dpi=144)
    plt.title('Comparison')
    plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    width = 0.4
    plt.bar([i - width for i in range(len(acc_names))], height=acc_values_NL2SQL, width=width, label='NL2SQL')
    plt.bar([i for i in range(len(acc_names))], height=acc_values_ATEP, width=width, label='ATEP')
    plt.bar([i - width / 2 for i in range(len(acc_names))], height=[0 for i in range(len(acc_names))],  tick_label=acc_names) # tick_label
    for i in range(0, len(acc_names)):
        plt.annotate(xy=(i - 1.6*width, acc_values_NL2SQL[i]), text=str(round(acc_values_NL2SQL[i], 1))+'%')
        plt.annotate(xy=(i - 0.5*width, acc_values_ATEP[i]), text=str(round(acc_values_ATEP[i], 1))+'%')
    # label and save
    plt.legend()
    plt.savefig('fig/comparison.png')
    return

def plot_bar0(agg_toks, NLQ_nums):
    fig = plt.figure(figsize=(10, 4), dpi=144)
    # percentage of NLQs with special tok
    c0 = 0
    x_all = []
    x_label = []
    for i in range(1, len(agg_ops)):
        height = [tok[1] / NLQ_nums[i] * 100 for tok in agg_toks[i]]
        labels = [tok[0] for tok in agg_toks[i]]
        width = 0.6
        x = [c0 + i for i in range(len(height))]
        plt.bar(x=x, height=height, width=width, label=agg_ops[i])
        for j in range(len(labels)):
            plt.annotate(xy=(c0 + j - width / 2, height[j]), text=str(round(height[j], 1)) + '%')
        c0 += len(height)
        x_all += x
        x_label += labels
    # possibility of AGG when speical tok in NLQ
    plt.xticks(x_all, x_label)
    plt.legend()
    plt.savefig('fig/agg_rl_stcs_freq.png')

def plot_bar(agg_toks):
    agg_toks = agg_toks[1:]
    fig = plt.figure(figsize=(12, 6), dpi=144)
    width = 0.8
    x_cur = 0
    x_all = []
    x_label = []
    for idx, agg_tok in enumerate(agg_toks):
        x = [x_cur + i for i in range(0, len(agg_tok))]
        height = [100 * tok[2] / (tok[2] + tok[3]) for tok in agg_tok]
        plt.bar(x=x, height=height, width=width, label=agg_ops[idx+1])
        for i in range(0, len(x)):
            plt.annotate(xy=(x[i] - width / 2, height[i]), text=str(round(height[i], 1)) + '%')
        x_cur += len(agg_tok)
        x_all += x
        x_label += [tok[0] for tok in agg_tok]
    plt.xticks(x_all, x_label)
    plt.legend()
    plt.savefig('fig/agg_rl_stcs.png')


def plot_stcs():
    res_path = 'data_and_model/agg_rl_toks.json'
    if len(sys.argv) == 2:
        res_path = sys.argv[1]
    with open(res_path, 'r') as f:
        [agg_toks, nums, NLQ_nums] = json.load(f)
    plot_bar0(agg_toks, NLQ_nums)
    plot_bar(agg_toks)
    return

if __name__ == "__main__":
    plot_one_res()
    # plot_acc_detail_compare()
    # plot_stcs()
