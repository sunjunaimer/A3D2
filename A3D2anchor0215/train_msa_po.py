import os
os.environ['CURL_CA_BUNDLE'] = ''  #for transformer error
import time
import numpy as np
from opts.get_opts import Options
from data import create_dataset_with_args
from models import create_model
from utils.logger import get_logger, ResultRecorder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from datetime import datetime
import torch
import random
import copy

import time

from models.networks.mybert import BertTop, BertTopSeq


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def tolist(data_loader):
    data_list = []
    for batch in data_loader:
        print('AAA')
        #batch = batch.to(device)
        data_list.append(batch)
    return data_list


def c2oh(class_numbers, num_classes=4):
    # Create a one-hot matrix
    one_hot_matrix = np.zeros((class_numbers.size, num_classes))
    one_hot_matrix[np.arange(class_numbers.size), class_numbers] = 1
    return one_hot_matrix

def eval_tr0806(model, val_iter, is_save=False, phase='test'):
    model.eval()
    total_pred = []
    total_label = []

    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        pred_f = c2oh(pred) * 1.1
        if 'A' in opt.modality:
            pred_A = model.pred_A.argmax(dim=1).detach().cpu().numpy()
            pred_f += c2oh(pred_A)
        if 'V' in opt.modality:
            pred_V = model.pred_V.argmax(dim=1).detach().cpu().numpy()
            pred_f += c2oh(pred_V)
        if 'L' in opt.modality:
            pred_L = model.pred_L.argmax(dim=1).detach().cpu().numpy()
            pred_f += c2oh(pred_L)
        
        pred_f = torch.tensor(pred_f)
        pred_f =  pred_f.argmax(dim=1)
        label = data['label']
        total_pred.append(pred_f)
        total_label.append(label)
        # del data, pred, label
        # torch.cuda.empty_cache()

    # calculate metrics
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average='weighted')
    f1 = f1_score(total_label, total_pred, average='weighted')
    cm = confusion_matrix(total_label, total_pred)
    model.train()

    # save test results
    if is_save:
        save_dir = model.save_dir
        np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
        np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

    return acc, uar, f1, cm


def eval(model, val_iter, is_save=False, phase='test'):
    model.eval()
    total_pred = []
    total_label = []

    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        label = data['label']
        total_pred.append(pred)
        total_label.append(label)
        # del data, pred, label
        # torch.cuda.empty_cache()

    # calculate metrics
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average='weighted')
    f1 = f1_score(total_label, total_pred, average='weighted')
    cm = confusion_matrix(total_label, total_pred)
    model.train()
    #model.netA = lora_wav2vec(model.netA)
    
    # save test results
    if is_save:
        save_dir = model.save_dir
        np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
        np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

    return acc, uar, f1, cm

    

##############################################################



def clean_chekpoints(expr_name, store_epoch):
    root = os.path.join(opt.checkpoints_dir, expr_name)
    for checkpoint in os.listdir(root):
        if not checkpoint.startswith(str(store_epoch)+'_') and checkpoint.endswith('pth'):
            os.remove(os.path.join(root, checkpoint))



if __name__ == '__main__':
    tic = time.time()
    f1_val = []
    f1_tst = []
    f1_trn = []
    loss_disc_trn = [] 
    loss_disc_val = [] 
    loss_disc_tst = [] 
    f1_t_val = []
    f1_t_tst = []
    f1_t_trn = []

    opt = Options().parse() 
    weights = np.array(opt.weights)                            # get training options   is_Train = True
    opt.weight = weights[opt.cof_weight-1]

    folder_path = os.path.join(opt.log_dir, opt.name)
    if not os.path.exists(folder_path):                 # make sure logger path exists
        os.mkdir(folder_path)    
    logger_path = os.path.join(folder_path, str(opt.cvNo)) # get logger path
    if not os.path.exists(logger_path):                 # make sure logger path exists
        os.mkdir(logger_path)

    total_cv = 100 
    result_recorder_s = ResultRecorder(os.path.join(opt.log_dir, opt.name, 'result_source.tsv'), total_cv=total_cv) # init result recoreder
    result_recorder_t = ResultRecorder(os.path.join(opt.log_dir, opt.name, 'result_target.tsv'), total_cv=total_cv)
    suffix = '_'.join([opt.model, opt.dataset_mode])    # get logger suffix
    logger = get_logger(logger_path, suffix)            # get logger
    if opt.has_test:
        opt.tmp_weight = opt.change_weight1
        opt.domain = 'source'                                    # create a dataset given opt.dataset_mode and other options
        dataset, val_dataset, tst_dataset = create_dataset_with_args(opt, dataset_name=opt.source, set_name=['train', 'val', 'test']) 

        opt.tmp_weight = opt.change_weight2
        opt.domain = 'target'
        dataset_t, val_dataset_t, tst_dataset_t = create_dataset_with_args(opt, dataset_name=opt.target, set_name=['train', 'val', 'test'])  # 建立目标域数据加载
        #opt.batch_size = bs
    else:
        dataset_t, val_dataset_t = create_dataset_with_args(opt, dataset_name=opt.target, set_name=['train', 'val'])

    dataset_size = len(dataset)    # get the number of images in the dataset.
    dataset_size_t = len(dataset_t)
    logger.info('The number of training samples = %d' % dataset_size)  
    logger.info('The number of training target samples = %d' % dataset_size_t)


    batch_step = int(len(dataset)/opt.batch_size + 1)


    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.cuda()
    total_iters = 0                # the total number of training iterations
    best_eval_acc, best_eval_uar, best_eval_f1 = 0, 0, 0
    best_eval_epoch = -1           # record the best eval epoch
    
    num_epoch = opt.niter + opt.niter_decay
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        start_steps = (epoch - 1) * batch_step
        total_steps = num_epoch * batch_step
        for i, (data, data_t) in enumerate(zip(dataset, dataset_t)):  # inner loop within one epoch 根据__iter__函数，每次传入一个batch数据的字典data data_t
            iter_start_time = time.time()   # timer for computation per iteration
            total_iters += 1                # opt.batch_size
            epoch_iter += opt.batch_size

            p = float(i + start_steps) / total_steps
            alpha = p ** 2
            

            model.set_input(data, data_t, alpha)           # unpack data from dataset and apply preprocessing
            model.optimize_parameters(epoch)   # calculate loss functions, get gradients, update network weights
                
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()  #
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                logger.info('Cur epoch {}'.format(epoch) + ' loss ' + 
                        ' '.join(map(lambda x:'{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(**losses))

            iter_data_time = time.time()



        if epoch % (opt.niter + opt.niter_decay) == 0:              # cache our model every <save_epoch_freq> epochs
            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            # model.save_networks('latest')  # delete
            model.save_networks(epoch)

        logger.info('End of training epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate(logger)                     # update learning rates at the end of every epoch.

    # test
    if opt.has_test:
        model.load_networks(opt.niter + opt.niter_decay)

        acc, uar, f1_fit, cm = eval(model, tst_dataset, is_save=True, phase='test')
        acc_t, uar_t, f1_fit_t, cm_t = eval(model, tst_dataset_t, is_save=True, phase='test')
        logger.info('Test results on source domain: acc %.4f uar %.4f f1 %.4f' % (acc, uar, f1_fit))
        logger.info('Test results on target domain: acc %.4f uar %.4f f1 %.4f' % (acc_t, uar_t, f1_fit_t))

        logger.info('\n{}'.format(cm))
        logger.info('\n{}'.format(cm_t))
        result_recorder_s.write_result_to_tsv({
            'acc': acc,
            'uar': uar,
            'f1': f1_fit,
            'weight': opt.weight,
        }, cvNo=opt.cvNo)

        result_recorder_t.write_result_to_tsv({
            'acc': acc_t,
            'uar': uar_t,
            'f1': f1_fit_t,
            'weight': opt.weight,
        }, cvNo=opt.cvNo)
    else:
        result_recorder_t.write_result_to_tsv({
            'acc': best_eval_acc,
            'uar': best_eval_uar,
            'f1': best_eval_f1
        }, cvNo=opt.cvNo)
    
    clean_chekpoints(opt.name + '/' + str(opt.cvNo), best_eval_epoch)

    toc = time.time()
    runtime = toc - tic 
    print('total running time: ', runtime)



