import numpy as np
from os import path as osp
from trainers.trainer import cls_ctcTrainer
from model import PlateNet
import torch
from torch import nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from config import opt
from datasets import data_manager
from datasets.data_loader import ImageData
from datasets.samplers import RandomIdentitySampler
from utils.serialization import Logger, save_checkpoint
from utils.transforms import TestTransform, TrainTransform

use_gpu = True

if __name__ == '__main__':

    dataset = data_manager.init_dataset(name=opt.dataset, mode=opt.mode)
    pin_memory = True if use_gpu else False
    summary_writer = SummaryWriter(osp.join(opt.save_dir, 'tensorboard_log'))

    trainloader = DataLoader(
        ImageData(dataset.train, TrainTransform(opt.datatype)),
        sampler=RandomIdentitySampler(dataset.train, opt.num_instances),
        batch_size=opt.train_batch, num_workers=opt.workers,
        pin_memory=pin_memory, drop_last=True
    )

    model = PlateNet(batch_size=opt.train_batch, n_class=66)
    optim_policy = model.get_optim_policy()
    if use_gpu:
        model = nn.DataParallel(model).cuda()
    print('dddddddddddd')

    criterion = nn.CTCLoss()

    # get optimizer
    if opt.optim == "sgd":
        optimizer = torch.optim.SGD(optim_policy, lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(optim_policy, lr=opt.lr, weight_decay=opt.weight_decay)

    start_epoch = opt.start_epoch
    # get trainer and evaluator
    plate_trainer = cls_ctcTrainer(opt, model, optimizer, criterion, summary_writer)

    def adjust_lr(optimizer, ep):
        if ep < 50:
            lr = 1e-4 * (ep // 5 + 1)
        elif ep < 200:
            lr = 1e-3
        elif ep < 300:
            lr = 1e-4
        else:
            lr = 1e-5
        for p in optimizer.param_groups:
            p['lr'] = lr

    # start training
    best_rank1 = opt.best_rank
    best_epoch = 0
    for epoch in range(start_epoch, opt.max_epoch):
        if opt.adjust_lr:
            adjust_lr(optimizer, epoch + 1)
        plate_trainer.train(epoch, trainloader)

        # skip if not save model
        if opt.eval_step > 0 and (epoch + 1) % opt.eval_step == 0 or (epoch + 1) == opt.max_epoch:
            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({'state_dict': state_dict, 'epoch': epoch + 1},
                is_best=False, save_dir=opt.save_dir,
                filename='checkpoint_ep' + str(epoch + 1) + '.pth.tar')

    print('Best rank-1 {:.1%}, achived at epoch {}'.format(best_rank1, best_epoch))


