import torch

import utility
import model
import numpy as np
import os
from collections import namedtuple
import imageio
from data import common
from option import args
import tqdm
import glob

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)


def forward_chop(*args, model = None, shave=10, min_size=160000):
    scale = 1
    # height, width
    h, w = args[0].size()[-2:]

    top = slice(0, h // 2 + shave)
    bottom = slice(h - h // 2 - shave, h)
    left = slice(0, w // 2 + shave)
    right = slice(w - w // 2 - shave, w)
    x_chops = [torch.cat([
        a[..., top, left],  # ...全面所有维度
        a[..., top, right],
        a[..., bottom, left],
        a[..., bottom, right]
    ]) for a in args]

    y_chops = []

    for i in range(0, 4):  # d=多GPU协作
        x = [x_chop[i:(i + 1)] for x_chop in x_chops]
        y = model(x[0]).detach().cpu()
        if not isinstance(y, list): y = [y]
        if not y_chops:
            y_chops = [[_y] for _y in y]
        else:
            for y_chop, _y in zip(y_chops, y):
                y_chop.extend(_y)


    h *= scale
    w *= scale
    top = slice(0, h // 2)
    # bottom = slice(h - h//2, h)
    bottom = slice(h // 2, h)
    # bottom_r = slice(h//2 - h, None)
    bottom_r = slice(h // 2 - h, None)
    left = slice(0, w // 2)
    # right = slice(w - w//2, w)
    right = slice(w // 2, w)
    # right_r = slice(w//2 - w, None)
    right_r = slice(w // 2 - w, None)

    # batch size, number of color channels
    b, c = y_chops[0][0].size()[:-2]
    y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
    for y_chop, _y in zip(y_chops, y):
        _y[..., top, left] = y_chop[0][..., top, left]
        _y[..., top, right] = y_chop[1][..., top, right_r]
        _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
        _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

    if len(y) == 1: y = y[0]

    return y


if checkpoint.ok:
    model = model.Model(args, checkpoint)  # model
    print("Model has {:.2f}M Parameters".format((np.sum([i.numel() for i in model.parameters()])) / 1.e6))
    model = model
    model.eval()
    checkpoint.done()

LR_path = args.dir_data
save_path = '../experiment/LR_result'

img_list = glob.glob(os.path.join(LR_path,'*.png'))


torch.set_grad_enabled(False)

for i in tqdm.tqdm(img_list, ncols=80):

    img = imageio.imread(i)
    lr = common.set_channel(img, n_channels=3)  # *解耦 从而可以传入*args
    lr = (common.np2Tensor(lr[0], rgb_range=255)[0]).unsqueeze(0).cuda()


    pred = model(lr,1)

    #detach 类似于原来的data，只不过现在tensor与variable合并，但是detach更安全（修改原tensor求导会报错）
    #先取到与tensor的data内容，再转cpu再转numpy

    pred = pred.detach().squeeze(0).permute(1,2,0).clamp(0,255).cpu().numpy()
    pred = pred.astype('uint8')
    imageio.imwrite(os.path.join(save_path,(i.split('/')[-1])),pred,format='png')
