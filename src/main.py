import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import numpy as np

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if args.data_test == 'video':
    from videotester import VideoTester
    model = model.Model(args, checkpoint)
    t = VideoTester(args, model, checkpoint)
    t.test()
else:
    if checkpoint.ok:
        loader = data.Data(args) #data 做数据集(dataset)的合并 并返回loader training合并 testing分开(list)
        model = model.Model(args, checkpoint) #model
        loss = loss.Loss(args, checkpoint) if not args.test_only else None #loss
        t = Trainer(args, loader, model, loss, checkpoint)
        print("Model has {:.2f}M Parameters".format((np.sum([i.numel() for i in model.parameters()]))/1.e6))
        while not t.terminate():
            t.train()#one epoch
            t.test()

        checkpoint.done()

