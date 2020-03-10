'''
   Copyright 2020 Sailist

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''
from collections import OrderedDict
from itertools import cycle, chain

from torch.utils.data import DataLoader
import torch

class DataBundler:
    """
    当一个模型需要训练多个数据集的时候，通过DataBundler类和附属的装饰器灵活的提供数据集::

        bundler = DataBundler() \
                .cycle(cifar_dataloader, "cifar") \
                .add(svhn_dataloader, "svhn").chain_iter()

        for (imgs,labels) in bundler:
            ...

    效果等价于::

        for (imgs,chains) in chain(cifar_dataloader,svhn_dataloader):
            ...

    主要方便Trainer使用，单独使用可能不会很方便？
    """

    def __init__(self):
        self.dataloaders = OrderedDict()
        self.iter_mode = "chain"

    def cycle(self, loader: DataLoader, name=None):
        """一般在zip中保证数据量少的数据集不会成为拖累"""
        assert isinstance(loader, DataLoader)
        self._append(loader, cycle, name)
        return self

    def _append(self, loader, func, name):
        if name is None:
            unname = "unnamed"
            i = 0
            name = "{}_{}".format(unname, i)
            while name in self.dataloaders:
                i += 1
                name = "{}_{}".format(unname, i)
        else:
            assert name not in self.dataloaders, "{} also defined in bundler".format(name)

        self.dataloaders[name] = (loader, func)

    def __len__(self):
        return sum(self.len_list())

    def len_list(self):
        """
        按照添加的顺序返回各个dataloader的长度（batch级别）
        :return:
        """
        return [len(loader) for _, (loader, _) in self.dataloaders.items()]

    def len_dict(self):
        """
        返回每个loader的 name:len 字典
        :return: an OrderedDict
        """
        res = OrderedDict()
        for name, (loader, func) in self.dataloaders.items():
            res[name] = len(loader)
        return res

    def add(self, loader, name=None):
        assert isinstance(loader, DataLoader)
        self._append(loader, lambda x: x, name)
        return self

    def func_loader(self):
        return [func(loader) for name, (loader, func) in self.dataloaders.items()]

    def __getitem__(self, item):
        return self.dataloaders[item][0]

    def __iter__(self):
        loaders = self.func_loader()
        if len(loaders) == 1:
            return iter(loaders[0])

        if self.iter_mode == "zip":
            return zip(*loaders)
        elif self.iter_mode == "chain":
            return chain(*loaders)

        assert False

    def choice_batch(self)->tuple:
        return next(iter(self))

    def choice_sample(self)->tuple:
        xs,ys = next(iter(self)) # type:(torch.Tensor,torch.Tensor)
        return (xs[0],ys[0])

    def zip_iter(self):
        """切换为zip方式提供所有添加的数据集"""
        self.iter_mode = "zip"
        return self

    def chain_iter(self):
        """
        切换为chain方式提供所有添加的数据集
            注意，如果以cycle方法添加了某个数据集，那么该迭代将永远不会停止
        :return:
        """
        self.iter_mode = "chain"
        return self
