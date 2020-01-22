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
   
    温馨提示：
        抵制不良代码，拒绝乱用代码。

        注意自我保护，谨防上当受骗。

        适当编程益脑，沉迷编程伤身。

        合理安排时间，享受健康生活！
'''
from functools import wraps
from itertools import chain
from queue import PriorityQueue
from typing import Iterable

import torch,os
from torch import nn
from torch.optim import optimizer
from torch.utils.data import DataLoader

from ..base.meta import Merge
from ..base.structure import WalkDict
from ..cacu import accuracy as acc
from .logwrapper import Logger
from .parameter import TrainParam,LogMeter
from .databundler import DataBundler
from .saver import Saver
import warnings

class BaseTrainer(metaclass=Merge):
    _ignore_call_back = {"model_dict", "optim_dict",
                         "model_state_dict", "optim_state_dict",
                         "create_checkpoint_dict", "create_checkpoint_dict",
                         "iter_train_dataloader", "iter_eval_dataloader", "iter_test_dataloader",
                         "predict", "preprocess",
                         "add_callback"}

    def __init__(self, param: TrainParam):
        # self.param = param
        self._param = param
        self._base_dir = "./release/"
        self.train_epoch_toggle = False
        self.train_toggle = False
        self.logger = Logger()


    def set_saver(self,path=None,max_to_keep=3):
        if path is None:
            path = os.path.join(self._base_dir,self._param.get_exp_name())
        self.saver = Saver(path,max_to_keep=max_to_keep)
        self.logger.line("Set Saver in {}".format(path))


    def add_log_path(self,log_dir = None):
        if log_dir is None:
            log_dir = self._base_dir
        fn = os.path.join(log_dir,self._param.get_exp_name(),"log.txt")
        self.logger.add_pipe(fn)

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)

        def wrapper(func, call_dict: dict):
            @wraps(func)
            def newfunc(*args, **kwargs):
                que = call_dict.setdefault(func.__name__, PriorityQueue())
                for callback in que.queue:
                    callback.on_begin(self, func, self._param, *args, **kwargs)
                meter = func(*args, **kwargs)
                for callback in que.queue:
                    callback.on_end(self, func, self._param, meter, *args, **kwargs)
                return meter

            return newfunc

        callback_dict = {}
        vars = dir(self)
        for name in vars:
            value = getattr(self, name)
            # for name, value in inspect.getmembers(self):
            if name.startswith("_"):
                continue
            if callable(value):
                callback_dict.setdefault(name, PriorityQueue())
                setattr(self, name, wrapper(value, callback_dict))
        self._callback_set = set()
        self._callback_dict = callback_dict
        return self

    def add_callback(self, func, callback):
        """
        添加一个回调函数
        :param func: trainer中的函数，建议用 trainer.function 的形式传入
        :type callable,str
        :param callback:
        :return:
        """
        msg = None
        if isinstance(func, str):
            funcname = func
            func = getattr(self, funcname, None)

        if func is None:
            msg = "Function not found."
            callback.on_hook_failed(self, func, msg)
        elif not hasattr(func, "__name__"):
            msg = "This function is not native."
            callback.on_hook_failed(self, None, msg)
        elif not hasattr(self, func.__name__):
            msg = "do not has function: {}".format(func.__name__)
            callback.on_hook_failed(self, func, msg)
        elif func.__name__.startswith("_"):
            msg = "function must not startswith '_'"
            callback.on_hook_failed(self, func, msg)
        elif func.__name__ in self._ignore_call_back:
            msg = "This function is in the ignored list, so you can't add callback on it."
            callback.on_hook_failed(self, func, msg)

        if msg is not None:
            return False

        que = self._callback_dict.setdefault(func.__name__, PriorityQueue())
        que.put(callback)
        callback.on_hooked(self, func, self._param)
        return True

    # def set_saver(self, base_dir, max_to_keep=3):
    #     self._saver = Saver(base_dir, max_to_keep)
    #     return {"save_path"}

    def _regist_dataloader(self, dataloader):
        bundler = DataBundler()
        if isinstance(dataloader, DataLoader):
            bundler.add(dataloader, name="test")
        elif isinstance(dataloader, dict):
            for k, v in dataloader.items():
                assert isinstance(v, DataLoader)
                bundler.add(v, "test_".format(k))
        elif isinstance(dataloader, DataBundler):
            bundler = dataloader
        elif isinstance(dataloader, Iterable):
            for i in dataloader:
                assert isinstance(i, DataLoader)
                bundler.add(i)

        return bundler

    def regist_dataloader(self,train,eval,test):
        self.regist_train_dataloader(train)
        self.regist_eval_dataloader(eval)
        self.regist_test_dataloader(test)

    def regist_test_dataloader(self, dataloader):
        """import dataloaders for test, in test mode, it will all be tested"""
        bundler = self._regist_dataloader(dataloader)
        self.test_databundler = bundler

        meter = LogMeter()
        meter.updata(**bundler.len_dict())
        return meter

    def regist_eval_dataloader(self, dataloader):
        """import dataloaders for eval, in eval mode, it will all be eval"""
        bundler = self._regist_dataloader(dataloader)
        self.eval_databundler = bundler

        meter = LogMeter()
        meter.updata(**bundler.len_dict())
        return meter

    def regist_train_dataloader(self, dataloader):
        bundler = self._regist_dataloader(dataloader)
        self.train_databundler = bundler

        meter = LogMeter()
        meter.updata(**bundler.len_dict())
        return meter

    def iter_train_dataloader(self):
        return self.train_databundler

    def iter_eval_dataloader(self):
        return self.eval_databundler

    def iter_test_dataloader(self):
        return self.test_databundler

    @property
    def device(self):
        if self._device is None:
            warnings.warn("try to get device before called regist_device()")
        return self._device

    @device.setter
    def device(self,d):
        self._device = d

    def regist_device(self, device=None):
        if device is None:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self._device = torch.device(device)
        elif isinstance(device, torch.device):
            self._device = device
        else:
            assert False, "device type not support, need str or torch.device, but {}".format(device)

        self._model_to_device()

    def model_dict(self) -> WalkDict:
        return self._model_dict

    def optim_dict(self) -> WalkDict:
        return self._optim_dict

    def model_state_dict(self):
        return {k: v.state_dict() for k, v in chain(self.model_dict().walk_items())}

    def optim_state_dict(self):
        return {k: v.state_dict() for k, v in chain(self.optim_dict().walk_items())}

    def create_checkpoint_dict(self) -> dict:
        """和torch.save的逻辑一样，要保存多少变量直接写上即可"""
        return {
            "eidx": self._param.eidx,
            "global_step": self._param.global_step,
            **self.model_state_dict(),
            **self.optim_state_dict(),
        }

    def create_model_state_dict(self) -> dict:
        """
        默认返回一个{model_name:model_dict}，model_dict的每一个key表示一个需要保存的模型
            （是模型而不是state_dict()）

        类似optimizer 中的 param group 方法，每一个key对应一个要保存的模型
            如果要保存多个模型到一个文件中，那么该文件的key对应一个模型字典即可
        :return:
        """

        # return {self.model_name: {k: v.state_dict() for k, v in self.raw_model_dict.items()}}
        return {self._model_name: self.model_state_dict()}

    def _model_to_device(self):
        if self.device is None:
            return
        if self.model_dict() is None:
            return

        for k, v in self.model_dict().walk_items():  # type:(None,nn.Module)
            v.to(self.device)

    def regist_model_and_optim(self, exp_name=None, **kwargs):
        """
        声明在该Trainer中用到的模型和优化器，从而可以自动完成模型保存和恢复（如果没有特殊用途的话）
        （一切保存和恢复通过 state_dict() load_state_dict() 完成的变量）
        :param exp_name: 试验名称，用于保存模型，断点等作为前缀，如果为空，则指定为类名
        :param kwargs:
        :return:
        """
        if exp_name is None:
            exp_name = self.__class__.__name__

        self._model_name = exp_name

        if not self._model_dict:
            self._model_dict = WalkDict()
        if not self._optim_dict:
            self._optim_dict = WalkDict()

        for k, v in kwargs.items():
            if isinstance(v, nn.Module):

                self._model_dict[k] = v
            elif isinstance(v, optimizer.Optimizer):
                self._optim_dict[k] = v
            else:
                assert False, "not module or optimizer"

        self._model_to_device()

    def save_model(self):
        self._check_saver()
        meter = LogMeter()
        for model_group_name, models_dicts in self.create_model_state_dict().items():  # type: str,dict
            meter[model_group_name] = self.saver.save(model=models_dicts,
                                                      fn_prefix=model_group_name)
        return meter

    def change_mode(self, train=True):
        if train:
            for _, v in self.model_dict().walk_items():  # type:(None,nn.Module)
                v.train()
        else:
            for _, v in self.model_dict().walk_items():  # type:(None,nn.Module)
                v.eval()

    def load_model(self, path):
        self._check_saver()
        ckpt = self.saver.load(path)
        for k, v in self.model_dict().walk_items():  # type:(str,nn.Module)
            if isinstance(v, nn.Module):
                v.load_state_dict(ckpt[k])

    def _check_saver(self):
        if self.saver is None:
            self.set_saver()

    def save_checkpoint(self, **kwargs):
        """
        保存断点，断点要保存的内容从 checkpoint_dict() 方法中获取，通过saver自动保存至相应路径。
        如果没有必要，不需要重写该函数
        :param kwargs: 其他要附加到 checkpoint 中的信息
        :return:
        """
        self._check_saver()
        meter = LogMeter()
        ckpt_dict = self.create_checkpoint_dict()
        ckpt_dict.update(**kwargs)
        meter.ckpt_fn = self.saver.checkpoint(self._param.eidx, ckpt_dict)

        return meter

    def load_checkpoint(self, pointer=-1, not_exist_ok=False) -> int:
        """
        load instance checkpoint, and return trained epoch
        :param pointer:
        :return:
        """
        self._check_saver()
        try:
            ckpt = self.saver.restore(pointer)
        except BaseException as e:
            if not_exist_ok:
                print(e.__class__.__name__)
                return 0
            else:
                raise e
        self._param.eidx = ckpt["eidx"]
        self._param.global_step = ckpt["global_step"]
        for k, v in self.model_dict().walk_items():  # type: (str, nn.Module)
            v.load_state_dict(ckpt[k])
        for k, v in self.optim_dict().walk_items():
            v.load_state_dict(ckpt[k])

        return self.eidx

    def train(self):
        meter = None
        param = self._param
        for eidx in range(param.eidx, param.epoch + 1):
            self.change_mode(True)
            meter = self.train_epoch(eidx, param)
            self.change_mode(False)
            self.eval()

            if self.train_toggle:
                self.train_toggle = False
                break

            '''将测试结果也添加到checkpoint中'''
        self.test()
        return meter

    def train_epoch(self, eidx, param):
        meter = None
        for idx, data in enumerate(self.iter_train_dataloader()):
            meter = self.train_batch(eidx, idx, param.global_step,
                             data, self.device,
                             param)
            param.global_step += 1
            param.eidx = eidx
            param.idx = idx
            if self.train_epoch_toggle:
                self.train_epoch_toggle = False
                break
        return meter

    def train_batch(self, eidx, idx, global_step, data, device, param):
        """
        :param eidx: 第几个epoch
        :param idx: 当前epoch的步
        :param global_step: 全局步
        :param data: batch级的数据
        :param param: 该中的参数只能读，不能写
        :return: meter，需要日志输出的内容
        """
        raise NotImplementedError()

    def preprocess(self, xs: torch.Tensor, ys: torch.Tensor):
        if isinstance(xs, torch.Tensor):
            xs = xs.to(self.device)
        if isinstance(ys, torch.Tensor):
            ys = ys.to(self.device)
        return xs, ys

    def predict(self, xs):
        raise NotImplementedError()

    def test(self):
        return self._test_eval_logic(self.iter_test_dataloader(), self._param)

    def eval(self):
        return self._test_eval_logic(self.iter_eval_dataloader(), self._param)

    def _test_eval_logic(self, dataloader, param: TrainParam):
        raise NotImplementedError()

    def __getattr__(self, item):
        return None


class Trainer(BaseTrainer):
    def __init__(self, param: TrainParam):
        super().__init__(param)

    def train_batch(self, eidx, idx, global_step, data, device, param):
        raise NotImplementedError()

    def preprocess(self, xs, ys):
        return super().preprocess(xs, ys)

    def predict(self, xs):
        raise NotImplementedError()

    def _test_eval_logic(self, dataloader, param: TrainParam):
        with torch.no_grad():
            count_dict = LogMeter(int)
            for xs, labels in dataloader:
                xs, labels = xs.to(self.device), labels.to(self.device)
                xs, labels = self.preprocess(xs, labels)
                preds = self.predict(xs)
                total, topk_res = acc.classify(preds, labels, topk=param.topk)
                count_dict["total"] += total
                for i, topi_res in zip(param.topk, topk_res):
                    count_dict[i] += topi_res
        return count_dict
