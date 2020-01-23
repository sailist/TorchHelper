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

    用于Trainer 执行过程中各方法的回调，
'''
import inspect,traceback
from functools import wraps
import os
import re
from typing import Any

from .parameter import TrainParam, LogMeter
from .trainerv2 import BaseTrainer


class Callback():
    """
    基类

    除了基本的回调接口外，还实现了`auto_hook`和`reverse_hook`两个方法，
    用来将callback实现的方法绑定到trainer和将trainer所有可回调的函数绑定到callback中。

    因为绑定的方法只会调用on_begin()和on_end()，因此对于具体的方法需要进行判断进行方法的分流，或者不使用自动绑定，
    而是主动用trainer绑定::

        trainer.add_callback(func=trainer.train, callback=cb)

    """
    priority = 0

    def __new__(cls,*_,**__) -> Any:
        self = super().__new__(cls)

        def wrapper(func):

            @wraps(func)
            def on_hooked(*args,**kwargs):
                self.first = getattr(self, "first", True)
                if self.first:
                    self.on_first_hooked(*args,**kwargs)
                    self.first = False
                func(*args,**kwargs)

            return on_hooked
        self.on_hooked = wrapper(self.on_hooked)
        return self

    def on_hooked(self, trainer: BaseTrainer, func, param: TrainParam):
        pass

    def on_first_hooked(self, trainer: BaseTrainer, func, param: TrainParam):
        """第一次绑定时调用，由元类控制，不受on_hooked等重写逻辑控制"""
        pass

    def on_hook_failed(self, trainer, func, message):
        pass

    def on_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        pass

    def on_exception(self,trainer:BaseTrainer,func,param:TrainParam,e:BaseException,*args,**kwargs):
        return False

    def on_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        pass

    def __le__(self, other):
        return self.priority <= other.priority

    def __eq__(self, other) -> bool:
        return self.priority == other.priority

    def __lt__(self, other):
        return self.priority < other.priority

    def reverse_hook(self, trainer: BaseTrainer):
        "将trainer 的所有可绑定的方法进行绑定"
        members = dir(trainer)
        for name in members:
            value = getattr(trainer, name)
            if not name.startswith("_") and callable(value):
                trainer.add_callback(name, self)

    def auto_hook(self, trainer: BaseTrainer):
        """自动将自己已有的on_func_begin/on_func_end方法绑定"""
        hookfunc_tmp = re.compile("on_(.*)_begin")
        members = inspect.getmembers(self)
        for name, value in members:
            match = re.search(hookfunc_tmp, name)
            if match:
                funcname = match.group(1)
                trainer.add_callback(funcname, self)


class TrainCallback(Callback):
    """
    实现了一般训练过程中的函数函数回调
    """

    def on_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        if func.__name__ == "train":
            self.on_train_begin(trainer, func, param, *args, **kwargs)
        elif func.__name__ == "train_epoch":
            self.on_train_epoch_begin(trainer, func, param, *args, **kwargs)
        elif func.__name__ == "train_batch":
            self.on_train_batch_begin(trainer, func, param, *args, **kwargs)
        elif func.__name__ == "test":
            self.on_test_begin(trainer, func, param, *args, **kwargs)
        elif func.__name__ == "eval":
            self.on_eval_begin(trainer, func, param, *args, **kwargs)

    def on_train_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        pass

    def on_train_epoch_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        pass

    def on_test_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        pass

    def on_eval_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        pass

    def on_train_batch_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        pass

    def on_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        if func.__name__ == "train":
            self.on_train_end(trainer, func, param, meter, *args, **kwargs)
        elif func.__name__ == "train_epoch":
            self.on_train_epoch_end(trainer, func, param, meter, *args, **kwargs)
        elif func.__name__ == "train_batch":
            self.on_train_batch_end(trainer, func, param, meter, *args, **kwargs)
        elif func.__name__ == "test":
            self.on_test_end(trainer, func, param, meter, *args, **kwargs)
        elif func.__name__ == "eval":
            self.on_eval_end(trainer, func, param, meter, *args, **kwargs)

    def on_train_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        pass

    def on_train_epoch_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        pass

    def on_test_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        pass

    def on_eval_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        pass

    def on_train_batch_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        pass


class DispatchCallback(Callback):
    """
    实现了任意函数的回调，子类实现的任意以`on_xxx_end`命名的函数，只要能够和Trainer中的`xxx`对应，都能够完成回调。

    可以参考:class DebugCallback
    """

    def on_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        call_name = "on_{}_begin".format(func.__name__)
        if hasattr(self, call_name):
            return getattr(self, call_name)(trainer, func, param, *args, **kwargs)

    def on_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        call_name = "on_{}_end".format(func.__name__)
        if hasattr(self, call_name):
            return getattr(self, call_name)(trainer, func, param, meter, *args, **kwargs)


class DrawCallBack(Callback):
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def on_hooked(self, trainer: BaseTrainer, func, param: TrainParam):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(os.path.join(self.base_dir, param.get_exp_name(), "board"))


class DebugCallback(DispatchCallback):
    """
    用于debug的回调函数，会绑定所有Trainer中能绑定的函数，并进行输出::

        dbg = DebugCallback()
        dbg.reverse_hook(trainer)

    """

    def on_hooked(self, trainer: BaseTrainer, func, param: TrainParam):
        trainer.logger.info(LogMeter(), "debug hooked on {}".format(func.__name__))

    def on_hook_failed(self, trainer, func, message):
        trainer.logger.info(LogMeter(), "debug {} failed".format(func))

    def on_exception(self, trainer: BaseTrainer, func, param: TrainParam, e: BaseException, *args, **kwargs):
        trainer.logger.info(LogMeter(), "debug catched exception when {}.".format(func))
        return super().on_exception(trainer, func, param, e, *args, **kwargs)

    def on_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        trainer.logger.info(LogMeter(), "debug {} start".format(func.__name__))

    def on_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        trainer.logger.info(meter, "debug {} end".format(func.__name__))


class ModelCheckpoint(TrainCallback):
    """
    用于自动保存模型断点::

        sv = ModelCheckpoint(save_path, monitor="all_loss", max_to_keep=3, mode="min")
        sv.auto_hook(trainer)

    """

    def __init__(self, base_dir, monitor, max_to_keep=3, mode="min"):
        self.base_dir = base_dir
        self.monitor = monitor
        self.max_to_keep = max_to_keep
        self.mode = mode

    def on_first_hooked(self, trainer: BaseTrainer, func, param: TrainParam):
        self.ckpt_dir = os.path.join(self.base_dir, param.get_exp_name())
        trainer.logger.info(prefix="ModelCheckpoint hooked.")
        trainer.set_saver(self.ckpt_dir, max_to_keep=self.max_to_keep)

    def on_exception(self, trainer: BaseTrainer, func, param: TrainParam, e: BaseException, *args, **kwargs):
        if not isinstance(e,KeyboardInterrupt):
            return False

        trainer.logger.newline()
        trainer.logger.info(prefix="Catched Exception.")
        ckpt_dict = trainer.create_checkpoint_dict()
        ckpt_dict.update(dict(
            _interrupt_info=traceback.format_exc(),
            _exception_class=e.__class__.__name__
        ))
        ckpt_fn = trainer.saver.checkpoint(param.eidx, ckpt_dict)
        trainer.logger.line("Model saved in {}".format(ckpt_fn))

        return False

    def on_train_epoch_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        if not hasattr(self, "monitor_var"):
            self.monitor_var = meter[self.monitor]

            trainer.logger.newline()
            trainer.logger.info(prefix="Model improved from nan to {}".format(self.monitor_var))
            ckpt_dict = trainer.create_checkpoint_dict()
            ckpt_dict.update(meter.logdict())
            ckpt_fn = trainer.saver.checkpoint(param.eidx, ckpt_dict)
            trainer.logger.info(LogMeter(), "Model saved in {}".format(ckpt_fn))

        elif (self.mode == "max" and meter[self.monitor] > self.monitor_var) or \
                (self.mode == "min" and meter[self.monitor] < self.monitor_var):
            trainer.logger.newline()
            trainer.logger.info(prefix="Model improved from {} to {}".format(meter[self.monitor],
                                                                             self.monitor_var))
            self.monitor_var = meter[self.monitor]
            ckpt_dict = trainer.create_checkpoint_dict()
            ckpt_dict.update(meter.logdict())
            ckpt_fn = trainer.saver.checkpoint(param.eidx, ckpt_dict)
            trainer.logger.info(LogMeter(), "Model saved in {}".format(ckpt_fn))

    def on_train_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        state_dict = trainer.create_model_state_dict()
        cmeter = LogMeter()
        for model_group_name, models_dicts in state_dict.items():  # type: str,dict
            cmeter[model_group_name] = trainer.saver.save(model=models_dicts,
                                                       fn_prefix=model_group_name)

        trainer.logger.info(cmeter, "Model Saved")


class Traininfo(TrainCallback):
    """用于实时输出模型训练信息"""
    def on_train_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        trainer.logger.info(LogMeter(), "Train start.")

    def on_test_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        trainer.logger.newline()
        trainer.logger.info(prefix="Test Start")

    def on_eval_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        trainer.logger.newline()
        trainer.logger.info(prefix="Eval Start")

    def on_train_epoch_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        # trainer.logger.newline()
        trainer.logger.info(meter, "Train epoch end:")

    def on_test_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        trainer.logger.info(meter, "Test end:")

    def on_eval_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        trainer.logger.info(meter, "Eval end:")

    def on_train_batch_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        trainer.logger.inline(meter, "Train Batch:")


class EarlyStop(TrainCallback):
    """用于控制模型提前停止 TODO 目前只用来临时使用过，内部逻辑没有完成"""
    def on_hooked(self, trainer: BaseTrainer, func, param: TrainParam):
        pass

    def on_train_batch_end(self, trainer: BaseTrainer, func, param: TrainParam, meter, *args, **kwargs):
        pass
        if param.idx > 10:
            trainer.train_epoch_toggle = True
            trainer.train_toggle = True


class AutoDevice(TrainCallback):
    """自动分配GPU和CPU"""
    def __init__(self,cpu_final = False) -> None:
        """
        :param cpu_final:如果所有的gpu都不可用，是否使用cpu训练，默认为False
        """
        import torch
        devices = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(torch.device("cuda:{}".format(i)))
        if cpu_final:
            devices.append(torch.device("cpu"))
        self.devices = devices
        self.idx = 0

    def on_hooked(self, trainer: BaseTrainer, func, param: TrainParam):
        if func.__name__ == "regist_device":
            trainer.regist_device(self.devices[self.idx])
            param.auto_device = True

    def on_exception(self, trainer: BaseTrainer, func, param: TrainParam, e, *args, **kwargs):
        if not isinstance(e,RuntimeError):
            return False
        if "CUDA out of memory" not in str(e):
            return False
        trainer.logger.line("cuda:{} out of memory.".format(self.idx))
        self.idx +=1
        if self.idx == len(self.devices):
            trainer.logger.line("All devices out of memory.")
            return False

        trainer.regist_device(self.devices[self.idx])
        trainer.logger.line("Change to cuda:{}".format(self.idx))
        return True

    def auto_hook(self, trainer: BaseTrainer):
        trainer.add_callback(trainer.regist_device,self)
        trainer.add_callback(trainer.train_batch,self)