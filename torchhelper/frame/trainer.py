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

    Trainer致力于简化模型的训练、验证、测试、断点、保存等一系列过程
    只要遵循Trainer的构造逻辑，即可迅速完成一整个训练流程

    Trainer内部依托torch实现了较为通用的train、eval、test逻辑，并且提供了准确率统计等方法
    同时，Trainer通过包内的另外一个类Saver实现了模型的保存和断点保存

    以及，Trainer内部维护了 eidx 和 global_step 两个变量，用来帮助类内和类外获取训练的次数信息

    温馨提示：
        抵制不良代码，拒绝乱用代码。
        注意自我保护，谨防上当受骗。
        适当编码益脑，沉迷编码伤身。
        合理安排时间，享受健康生活！
'''
import traceback
import warnings
from collections import Iterable
from functools import wraps

import torch
import torch.nn as nn

from .logger import Logger, LogParam


def start_interrupt_protect(protect_func, *pargs, **pkwargs):
    def wrapper(func):
        @wraps(func)
        def inner(self, *args, **kwargs):
            try:
                func(self, *args, **kwargs)
            except KeyboardInterrupt:
                protect_func(*pargs, **pkwargs)

        return inner

    return wrapper


class TrainerParams:
    '''
    用于设置各种超参数，saver等不会变更的变量
    _default_fields 用于设置默认参数

    该类在构造的时候接受字典，同时内含各种默认参数
    '''
    _default_fields = dict(
        epoch=100,  # 训练的epoch次数
        topk=(1, 5),  # 求准确率的时候求topk的准确率
    )

    def __init__(self, **kwargs):
        for k, v in self.__class__._default_fields.items():
            kwargs.setdefault(k, v)
        for k, v in kwargs.items():
            if callable(v):
                v = v()
            setattr(self, k, v)

    def set_default(self, key, value=None):
        '''检查某值是否存在，如果不存在，则设置其默认值为value'''
        if not hasattr(self, key):
            if callable(value):
                value = value()
            setattr(self, key, value)

    def set_defaults(self, **kwargs):
        '''检查是否包含某个名字的变量，如果不包含，则设置其值为value'''
        for k, v in kwargs.items():
            self.set_default(k, v)

    def assert_params(self, *args):
        '''检查该实例化对象中是否包含某个名字的变量，如果不存在则抛出异常'''
        for k in args:
            assert hasattr(self, k), "{} must have param '{}'".format(self.__class__.__name__, k)

    @staticmethod
    def from_dict(params_dict: dict):
        '''从一个字典中构建一个TrainerParams对象'''
        return TrainerParams(**params_dict)


class Trainer():
    '''
        Trainer类实例构建的第一个参数即TrainerParam，
        实现了保存ckpt和model的逻辑，抽象了加载ckpt和model内容的方法，抽象了其他通用的train、eval、test、predict方法
        实现了logger模块的日志记录功能
    '''
    _essential_param = []
    _default_param_dict = {}

    def __init__(self, params):
        if isinstance(params, dict):
            params = TrainerParams.from_dict(params)
        self.params = params
        self.load_params()

    @property
    def model_name(self):
        if not hasattr(self, "__model_name"):
            _model_name = "auto_{}_model".format(self.__class__.__name__)
            warnings.warn("未使用 regeist_model_name() 方法指定方法名或重写 model_dict() 方法，保存模型名称前缀将使用 {}".format(_model_name))
        else:
            _model_name = self.__model_name

        return _model_name

    @property
    def raw_model_dict(self):
        overwrite_me = "\n".join([self.create_model_dict.__name__,
                                  self.load_model.__name__,
                                  self.create_checkpoint_dict.__name__,
                                  self.load_checkpoint.__name__])
        assert hasattr(self, "_model_dict"), \
            "需要使用 {} 方法指定模型与优化器，或重写以下方法：\n{}".format(
                self.regist_model_and_optim.__name__,
                overwrite_me)


        return self._model_dict

    def regist_model_and_optim(self, **kwargs):
        '''
        声明在该Trainer中用到的模型和优化器，从而可以自动完成模型保存和恢复（如果没有特殊用途的话）
        （一切保存和恢复通过 state_dict() load_state_dict() 完成的变量）
        :param kwargs:
        :return:
        '''
        if not hasattr(self, "__model_dict"):
            self._model_dict = dict()

        for k, v in kwargs.items():
            self._model_dict[k] = v

    def load_checkpoint(self, pointer=-1, not_exist_ok = False) -> int:
        '''
        load instance checkpoint, and return trained epoch
        :param pointer:
        :return:
        '''
        try:
            ckpt = self.saver.restore(pointer)
        except BaseException as e:
            if not_exist_ok:
                print(e.__class__.__name__)
                return 0
            else:
                raise e
        self.eidx = ckpt["eidx"]
        self.global_step = ckpt["global_step"]
        for k, v in self.raw_model_dict.items():
            v.load_state_dict(ckpt[k])

        return self.eidx

    def load_model(self, path):
        ckpt = self.saver.load(path)
        for k, v in self.raw_model_dict.items():
            if isinstance(v, nn.Module):
                v.load_state_dict(ckpt[k])

    def save_checkpoint(self, **kwargs):
        '''
        保存断点，断点要保存的内容从 checkpoint_dict() 方法中获取，通过saver自动保存至相应路径。
        如果没有必要，不需要重写该函数
        :param kwargs: 其他要附加到 checkpoint 中的信息
        :return:
        '''
        meter = LogParam()
        ckpt_dict = self.create_checkpoint_dict()
        ckpt_dict.update(**kwargs)
        meter.ckpt_fn = self.saver.checkpoint(self.eidx, ckpt_dict)
        return meter

    def save_model(self):
        meter = LogParam()
        for k, v in self.create_model_dict().items():
            meter[k] = self.saver.save(v, fn_prefix=k)
        return meter

    def create_checkpoint_dict(self) -> dict:
        '''和torch.save的逻辑一样，要保存多少变量直接写上即可'''
        return {
            "eidx": self.eidx,
            "global_step": self.global_step,
            **{k:v.state_dict() for k,v in self.raw_model_dict.items()}
        }

    def create_model_dict(self) -> dict:
        '''
        需要由子类实现，返回一个字典，每一个key表示一个需要保存的模型，该方法和load_model 方法对应
        因此可以任意实现，保证返回类型即可

        类似optimizer 中的 param group 方法，每一个key对应一个要保存的模型
            如果要保存多个模型到一个文件中，那么该文件的key对应一个模型字典即可
        :return:
        '''

        return {self.model_name: {k:v.state_dict() for k,v in self.raw_model_dict.items()}}

    def train(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def load_params(self):
        self.params.set_defaults(
            **self.__class__._default_param_dict
        )
        self.params.assert_params(*self.__class__._essential_param)
        return self._default_param_dict

    def log_handle(self, logger: Logger, fn, **kwargs):
        self.__setattr__(fn.__name__, logger.wrap(fn, **kwargs))

    @property
    def eidx(self):
        '''epoch'''
        return getattr(self, "_eidx", 0)

    @eidx.setter
    def eidx(self, value):
        self._eidx = value

    @property
    def global_step(self):
        '''global_step = old_epoch * total_step_in_epoch + current_step'''
        return getattr(self, "_global_step", 0)

    @global_step.setter
    def global_step(self, value):
        self._global_step = value

    def __getattr__(self, item):
        # if isinstance(self.params, dict):
        #     return self.params[item]
        return getattr(self.params, item)


class NormalTrainer(Trainer):
    '''
        实现了正常训练器的训练、验证、测试的逻辑
        仍然是抽象类，需要子类实现_train_epoch、predict等方法
        求accuracy等的方法，虽然可以放到类外，但因为和一些参数输出相关，因此放在类内部
    '''
    _essential_param = ["saver"]

    def __init__(self, params, eval_dataloader, test_dataloader, device=None):
        super().__init__(params)

        self.test_dataloader = test_dataloader
        self.eval_dataloader = eval_dataloader

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            assert False, "device type not support, need str or torch.device, but {}".format(device)

    def accuracy(self, preds, labels, cacu_rate=False):
        k = self.topk
        _, maxk = torch.topk(preds, max(*k), dim=-1)
        total = labels.size(0)
        test_labels = labels.view(-1, 1)  # reshape labels from [n] to [n,1] to compare [n,k]

        if cacu_rate:
            return [(test_labels == maxk[:, 0:i]).sum().item() / total for i in k]
        else:
            return total, [(test_labels == maxk[:, 0:i]).sum().item() for i in k]

    def train_epoch(self, eidx):
        raise NotImplementedError()

    def train(self):
        self.base_train()

    def train_from_check_point(self, pointer=-1):
        self.load_checkpoint(pointer)
        self.base_train()

    def safe_train(self, unsafe_exit=False):
        '''
        安全的训练模型，如果出现各种异常则保存断点
        :param unsafe_exit: 如果遇到了异常，处理完后是否直接退出程序（异常状态 1 ）
        :return: 如果不直接退出程序，那么返回是否执行成功，处理过异常返回False，未处理过异常返回True

        '''
        try:
            self.base_train()
            return True
        except (Exception, KeyboardInterrupt) as e:
            self.save_checkpoint(_interrupt_info=traceback.format_exc(),
                                 _exception_class=e.__class__.__name__)
            print()
            if unsafe_exit:
                exit(1)
        return False

    def base_train(self):
        for eidx in range(self.eidx, self.epoch + 1):
            self.eidx = eidx
            res = self.train_epoch(eidx)

            # if cancel the logger, the train_epoch() function may return Iterable object because of 'yield'
            if isinstance(res, Iterable):
                for i in res:
                    pass

            self.save_checkpoint()
            self.eval()
        self.test()
        self.save_model()

    def predict(self, x):
        raise NotImplementedError()

    def preprocess(self, x, y):
        return x, y

    def _test_eval_logic(self, dataloader):
        with torch.no_grad():
            count_dict = LogParam(int)
            for xs, labels in dataloader:
                xs, labels = xs.to(self.device), labels.to(self.device)
                xs, labels = self.preprocess(xs, labels)
                preds = self.predict(xs)
                total, topk_res = self.accuracy(preds, labels)
                count_dict["total"] += total
                for i, topi_res in zip(self.topk, topk_res):
                    count_dict[i] += topi_res
        return count_dict

    def test(self):
        count_dict = self._test_eval_logic(self.test_dataloader)
        return count_dict

    def eval(self):
        count_dict = self._test_eval_logic(self.eval_dataloader)
        return count_dict


class SemiTrainer(NormalTrainer):
    '''
        因为半监督学习需要同时传入有监督数据和无监督数据，因此与NormalTrainer相比
        该子类只多了两个数据集的参数
    '''

    def __init__(self, params, sup_dataloader, unsup_dataloader, eval_dataloader, test_dataloader, device):
        super().__init__(params, eval_dataloader, test_dataloader, device)
        self.sup_dataloader = sup_dataloader
        self.unsup_dataloader = unsup_dataloader


class SupervisedTrainer(NormalTrainer):
    '''
        有监督学习只需要传入有监督数据
    '''

    def __init__(self, params, sup_dataloader, eval_dataloader, test_dataloader, device):
        super().__init__(params, eval_dataloader, test_dataloader, device)
        self.sup_dataloader = sup_dataloader


class SingleModelTrainer(SupervisedTrainer):
    '''
        只有一个模型的最通用的Trainer
        from torchhelper.trainer import SingleModelTrainer
        model = get_model(...)
        params = TrainerParam()
        optimizer = ...
        train_dataloader = ...
        eval_dataloader = ...
        test_dataloader = ...
        trainer = SingleModelTrainer(
                        params,model,optimizer,loss = "mse",
                        train_dataloader,eval_dataloader,test_dataloader,
                        logged = True)

        trainer.train()

    '''
    _loss_dict = {
        "mse": nn.MSELoss(),
        "cross_entropy": nn.CrossEntropyLoss(),
    }

    def __init__(self, params, model: nn.Module, optimizer, loss, sup_dataloader, eval_dataloader, test_dataloader,
                 device, logged=True):
        super().__init__(params, sup_dataloader, eval_dataloader, test_dataloader, device)
        self.model = model
        self.optimizer = optimizer
        # TODO 分 str、fn 两类做判断
        if isinstance(loss, str):
            assert loss in SingleModelTrainer._loss_dict, "can't reco the loss"
            self.lossfn = SingleModelTrainer._loss_dict[loss]
        elif callable(loss):
            self.lossfn = loss
        else:
            assert False, "loss must be str or callable, but get %s" % loss

        self.regist_model_name(self.model.__class__.__name__)
        self.regist_model_and_optim(model=self.model, optim=self.optimizer)

        if logged:
            self.logger = Logger()
            self.log_handle(self.logger, self.train, prefix="Train model: ")
            self.log_handle(self.logger, self.test, prefix="Test  model: ")
            self.log_handle(self.logger, self.eval, prefix="Eval  model: ")
            self.log_handle(self.logger, self.save_checkpoint, prefix="Save Checkpoint: ")
            self.log_handle(self.logger, self.save_model, prefix="Save model: ")
            self.log_handle(self.logger, self.train_epoch, inline=True, prefix="Save model: ", cacu_step=True)

    def predict(self, x):
        return self.model(x)

    def train_epoch(self, eidx):
        for idx, (xs, labels) in enumerate(self.sup_dataloader):
            logits = self.model(xs)
            loss = self.lossfn(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
