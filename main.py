from transformers import get_scheduler
from functions import *

if __name__ == '__main__':

    ## 1. Hyper parameters
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)

    ## 2. Paths
    path_h = './data_and_model'  # '/home/wonseok'
    path_wikisql = './data_and_model'  # os.path.join(path_h, 'data', 'wikisql_tok')
    BERT_PT_PATH = path_wikisql

    path_save_for_evaluation = './'

    ## 3. Load data

    train_table, dev_table, train_loader, dev_loader = get_data(path_wikisql, args) # train_data and dev_data not used
    # test_data, test_table = load_wikisql_data(path_wikisql, mode='test', toy_model=args.toy_model, toy_size=args.toy_size, no_hs_tok=True)
    # test_loader = torch.utils.data.DataLoader(
    #     batch_size=args.bS,
    #     dataset=test_data,
    #     shuffle=False,
    #     num_workers=4,
    #     collate_fn=lambda x: x  # now dictionary values are not merged!
    # )
    ## 4. Build & Load models
    if not args.trained:
        model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH)
        results_for_plot = []
    else:
        # To start from the pre-trained models, un-comment following lines.
        path_model_bert = './model_bert_best.pt'
        path_model = './model_best.pt'
        model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH, trained=True,
                                                               path_model_bert=path_model_bert, path_model=path_model)
        with open('res/results_for_plot.json', 'r') as f:
            results_for_plot = json.load(f)

    ## 5. Get optimizers
    if args.do_train == 1:
        opt, opt_bert = get_opt(args.lr, args.lr_bert, model, model_bert, args.fine_tune)
        scheduler, scheduler_bert = get_scheduler(opt, opt_bert)

        ## 6. Train
        acc_lx_t_best = -1
        acc_best = [0, 0, 0, 0, 0, 0]
        epoch_best = -1
        print('Execution Guided Decoding:', args.EG)
        print("total epoch=", args.tepoch, " | ", "train_size=", args.train_size, " | ","test_size=", args.test_size)
        
        for epoch in range(args.tepoch):
            # train
            acc_train=None
            acc_train, aux_out_train = train(train_loader,
                                             train_table,
                                             model,
                                             model_bert,
                                             opt,
                                             bert_config,
                                             tokenizer,
                                             args.max_seq_length,
                                             args.num_target_layers,
                                             args.accumulate_gradients,
                                             opt_bert=opt_bert,
                                             st_pos=0,
                                             path_db=path_wikisql,
                                             dset_name='train',
                                             train_size=args.train_size)
            scheduler.step()
            scheduler_bert.step()
            if acc_train!=None:
              print_result(epoch, acc_train, 'train')
            # check DEV
            with torch.no_grad():
                acc_dev, results_dev, cnt_list = test(dev_loader,
                                                      dev_table,
                                                      model,
                                                      model_bert,
                                                      bert_config,
                                                      tokenizer,
                                                      args.max_seq_length,
                                                      args.num_target_layers,
                                                      detail=False,
                                                      path_db=path_wikisql,
                                                      st_pos=0,
                                                      dset_name='test', EG=args.EG,
                                                      test_size=args.test_size)
            print_result(epoch, acc_dev, 'test')

            # 保存结果用于绘图
            save_epoch_for_plot(results_for_plot, acc_train, acc_dev)
            # save results for the official evaluation
            save_for_evaluation(path_save_for_evaluation, results_dev, 'dev')

            # save best model
            # Based on Dev Set logical accuracy lx
            acc_best = save_best_model(acc_dev, acc_best, model, model_bert)
            acc_lx_t = acc_dev[-2]
            if acc_lx_t > acc_lx_t_best:
                acc_lx_t_best = acc_lx_t
                epoch_best = epoch
                # save best model
                state = {'model': model.state_dict()}
                torch.save(state, os.path.join('.', 'model_best.pt'))

                state = {'model_bert': model_bert.state_dict()}
                torch.save(state, os.path.join('.', 'model_bert_best.pt'))

            print(f"========Best Dev lx acc: {acc_lx_t_best} at epoch: {epoch_best}========")
    else:
        with torch.no_grad():
            acc_dev, results_dev, cnt_list = test(dev_loader,
                                                  dev_table,
                                                  model,
                                                  model_bert,
                                                  bert_config,
                                                  tokenizer,
                                                  args.max_seq_length,
                                                  args.num_target_layers,
                                                  detail=False,
                                                  path_db=path_wikisql,
                                                  st_pos=0,
                                                  dset_name='test', EG=args.EG,
                                                  test_size=args.test_size)
        print_result(0, acc_dev, 'test')

    if args.do_infer:
        # To use recent corenlp: https://github.com/stanfordnlp/python-stanford-corenlp
        # 1. pip install stanford-corenlp
        # 2. download java crsion
        # 3. export CORENLP_HOME=/Users/wonseok/utils/stanford-corenlp-full-2018-10-05

        # from stanza.nlp.corenlp import CoreNLPClient
        # client = CoreNLPClient(server='http://localhost:9000', default_annotators='ssplit,tokenize'.split(','))

        import corenlp

        client = corenlp.CoreNLPClient(annotators='ssplit,tokenize'.split(','))

        nlu1 = "Which company have more than 100 employees?"
        path_db = './data_and_model'
        db_name = 'ctable'
        data_table = load_jsonl('./data_and_model/ctable.tables.jsonl')
        table_name = 'ftable1'
        n_Q = 100000 if args.infer_loop else 1
        for i in range(n_Q):
            if n_Q > 1:
                nlu1 = input('Type question: ')
            pr_sql_i, pr_ans = infer(
                nlu1,
                table_name, data_table, path_db, db_name,
                model, model_bert, bert_config, max_seq_length=args.max_seq_length,
                num_target_layers=args.num_target_layers,
                beam_size=1, show_table=False, show_answer_only=False
            )
