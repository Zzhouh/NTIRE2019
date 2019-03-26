from importlib import import_module
from dataloader import MSDataLoader
from torch.utils.data import ConcatDataset

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale) #设置每个数据集的缩放尺寸

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train: #是可以有多个数据集的,将对应数据集的dataset类拼接起来作为最后的训练数据
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                datasets.append(getattr(m, module_name)(args, name=d))#添加对应数据集的类

            self.loader_train = MSDataLoader(
                args,
                MyConcatDataset(datasets),#合并
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu
            )

        self.loader_test = []
        if not args.not_test:
            for d in args.data_test: #同上 可以输入多个数据集,最后拼接起来
                if d in ['Set5', 'Set14', 'B100', 'Urban100']:
                    m = import_module('data.benchmark')
                    testset = getattr(m, 'Benchmark')(args, train=False, name=d)
                else:
                    module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                    m = import_module('data.' + module_name.lower())
                    print(module_name)
                    testset = getattr(m, module_name)(args, train=False, name=d)

                self.loader_test.append(MSDataLoader(
                    args,
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu
                ))

