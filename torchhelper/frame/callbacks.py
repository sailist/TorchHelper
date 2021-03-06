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
import inspect
import os
import re
import traceback
from functools import wraps
from typing import Any

import matplotlib.pyplot as plt
import torch

from .parameter import TrainParam, LogMeter, HistoryMeter
from .trainer import BaseTrainer


class Callback():
    """
    基类

    除了基本的回调接口外，还实现了`auto_hook`和`reverse_hook`两个方法，
    用来将callback实现的方法绑定到trainer和将trainer所有可回调的函数绑定到callback中。

    因为绑定的方法只会调用on_begin()和on_end()，因此对于具体的方法需要进行判断进行方法的分流，或者不使用自动绑定，
    而是主动用trainer绑定::

        trainer.add_callback(func=trainer.train, callback=cb)

    """
    priority = 0  # 所有内部实现的Callback优先级均在 [0-100] 以内

    def __new__(cls, *_, **__) -> Any:
        self = super().__new__(cls)
        self.enable = True

        def hook_wrap(func):
            """钩子第一次调用的时候执行"""

            @wraps(func)
            def on_hooked(*args, **kwargs):
                self.first = getattr(self, "first", True)
                if self.first:
                    self.on_first_hooked(*args, **kwargs)
                    self.first = False
                func(*args, **kwargs)

            return on_hooked

        def ecp_wrap(func):
            """同一个异常第一次调用的时候运行"""

            @wraps(func)
            def on_exception(trainer: BaseTrainer, tfunc, param: TrainParam, e: BaseException, *args, **kwargs):
                self.ecp = getattr(self, "ecp", None)
                if self.ecp != e:
                    self.on_first_exception(trainer, tfunc, param, e, *args, **kwargs)
                    self.ecp = e
                func(trainer, tfunc, param, e, *args, **kwargs)

            return on_exception

        def hook_failed_wrap(func):
            """钩子第一次调用的时候执行"""

            @wraps(func)
            def on_hook_failed(*args, **kwargs):
                self.first_failed = getattr(self, "first_failed", True)
                if self.first_failed:
                    self.on_first_hooked_failed(*args, **kwargs)
                    self.first_failed = False
                func(*args, **kwargs)

            return on_hook_failed

        self.on_hook_failed = hook_failed_wrap(self.on_hook_failed)
        self.on_hooked = hook_wrap(self.on_hooked)
        self.on_exception = ecp_wrap(self.on_exception)
        return self

    def on_hooked(self, trainer: BaseTrainer, func, param: TrainParam):
        pass

    def on_first_hooked(self, trainer: BaseTrainer, func, param: TrainParam):
        """第一次绑定时调用，由元类控制，不受on_hooked等重写逻辑控制"""
        trainer.logger.line("{} hooked on {}.".format(self, trainer))

    def on_first_exception(self, trainer: BaseTrainer, func, param: TrainParam, e: BaseException, *args, **kwargs):
        """第一次绑定时调用，由元类控制，不受on_exception等重写逻辑控制"""
        pass

    def on_first_hooked_failed(self, trainer, func, message):
        trainer.logger.line("{} hooked failed,msg={}".format(self, message))

    def on_hook_failed(self, trainer, func, message):
        pass

    def on_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        pass

    def on_exception(self, trainer: BaseTrainer, func, param: TrainParam, e: BaseException, *args, **kwargs):
        return False

    def on_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        pass

    def __le__(self, other):
        return self.priority <= other.priority

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

    def unhook(self):
        def NULLptr(*args, **kwargs):
            return False

        self.disposable = True
        for name in dir(self):
            value = getattr(self, name, None)
            if callable(value) and name.startswith("on_"):
                setattr(self, name, NULLptr)

    def _repr_by_val(self, *vals):
        vstr = "; ".join(["{}={}".format(val, str(getattr(self, val, None))) for val in vals])
        return "{}([{}])".format(self.__class__.__name__, vstr)

    def __repr__(self) -> str:
        return self._repr_by_val("priority")

    def toggle_enable(self, toggle=None):
        if toggle is not None:
            self.enable = toggle
        else:
            self.enable = not self.enable


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


class DrawCallBack(TrainCallback):
    def __init__(self, write_interval=50, path=None):
        self.path = path
        self.interval = write_interval

    def on_train_batch_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        if param.global_step % self.interval == self.interval - 1:
            for k, v in meter.board_dict().items():
                if type(v) in {int, float}:
                    trainer.writer.add_scalar(k, v)
                elif isinstance(v, torch.Tensor):
                    trainer.writer.add_scalar(k, v)

    def on_first_hooked(self, trainer: BaseTrainer, func, param: TrainParam):
        trainer.logger.info(prefix="{} hooked {}.".format(self.__class__.__name__, trainer.__class__.__name__))
        trainer.set_writter(self.path)

    def on_train_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        super().on_train_begin(trainer, func, param, *args, **kwargs)

    def on_train_epoch_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        super().on_train_epoch_begin(trainer, func, param, *args, **kwargs)

    def on_test_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        super().on_test_begin(trainer, func, param, *args, **kwargs)

    def on_train_batch_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        super().on_train_batch_begin(trainer, func, param, *args, **kwargs)

    def on_train_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        super().on_train_end(trainer, func, param, meter, *args, **kwargs)

    def on_train_epoch_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        trainer.writer.flush()

    def on_test_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        super().on_test_end(trainer, func, param, meter, *args, **kwargs)

    def on_eval_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        super().on_eval_end(trainer, func, param, meter, *args, **kwargs)


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

    def __init__(self, monitor, max_to_keep=3, mode="min", path=None):
        self.monitor = monitor
        self.path = path
        self.max_to_keep = max_to_keep
        self.mode = mode

    def on_first_hooked(self, trainer: BaseTrainer, func, param: TrainParam):
        super().on_first_hooked(trainer, func, param)
        trainer.set_saver(self.path, max_to_keep=self.max_to_keep)

    def on_first_exception(self, trainer: BaseTrainer, func, param: TrainParam, e: BaseException, *args, **kwargs):
        if not isinstance(e, KeyboardInterrupt):
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
        if not hasattr(self, "monitor_var") or \
                (self.mode == "max" and meter[self.monitor] > self.monitor_var) or \
                (self.mode == "min" and meter[self.monitor] < self.monitor_var):

            if not hasattr(self, "monitor_var"):
                self.monitor_var = "nan"

            trainer.logger.newline()
            trainer.logger.info(prefix="Model improved from {} to {}".format(self.monitor_var,
                                                                             meter[self.monitor]
                                                                             ))
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

    def on_eval_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        super().on_eval_end(trainer, func, param, meter, *args, **kwargs)

    def on_test_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        super().on_test_end(trainer, func, param, meter, *args, **kwargs)

    def __repr__(self) -> str:
        return self._repr_by_val("monitor", "max_to_keep", "mode", "path")


class ExpCheckpoint(TrainCallback):
    def __init__(self, per_epoch=100, start_epoch=0):
        self.per_epoch = per_epoch
        self.start_epoch = start_epoch

    def on_first_hooked(self, trainer: BaseTrainer, func, param: TrainParam):
        super().on_first_hooked(trainer, func, param)
        assert trainer.saver is not None, "Please call set_saver() to set a saver"

    def on_train_epoch_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        if (param.eidx - 1) % self.per_epoch == self.per_epoch - 1 and self.start_epoch < (param.eidx - 1):
            ckpt_dict = trainer.create_checkpoint_dict()
            ckpt_dict.update(meter.logdict())
            ckpt_dict["eidx"] += 1
            ckpt_fn = trainer.saver.check_keyepoch(param.eidx, ckpt_dict)
            trainer.logger.info(LogMeter(), "Model saved in {}".format(ckpt_fn))

    def on_eval_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        if (param.eidx - 1) % self.per_epoch == self.per_epoch - 1 and self.start_epoch < (param.eidx - 1):
            ckpt_fn = trainer.saver.append_info_to_keyepoch(param.eidx, meter.logdict())
            trainer.logger.info(LogMeter(), "Eval Result Append to {}".format(ckpt_fn))

    def __repr__(self) -> str:
        return self._repr_by_val("per_epoch","start_epoch")


class Traininfo(TrainCallback):
    """用于实时输出模型训练信息"""

    def on_first_hooked(self, trainer: BaseTrainer, func, param: TrainParam):
        super().on_first_hooked(trainer, func, param)

    def on_train_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        trainer.logger.line("Train start")
        trainer.logger.info(param, "With Param:")
        trainer.logger.info(LogMeter().merge_dict(trainer.train_databundler.len_dict()),
                            "With TrainDataset:")
        trainer.logger.info(LogMeter().merge_dict(trainer.test_databundler.len_dict()),
                            "With TestDataset:")
        trainer.logger.info(LogMeter().merge_dict(trainer.eval_databundler.len_dict()),
                            "With EvalDataset:")

    def on_test_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        trainer.logger.newline()
        trainer.logger.line("Test Start")

    def on_eval_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        trainer.logger.newline()
        trainer.logger.line("Eval Start")

    def on_train_epoch_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        # trainer.logger.newline()
        trainer.logger.info(meter, "Train epoch end:")

    def on_test_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        trainer.logger.info(meter, "Test end:")

    def on_eval_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        trainer.logger.info(meter, "Eval end:")

    def on_train_batch_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        trainer.logger.inline(meter, "Train Batch:")

    def on_train_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        trainer.logger.inline(meter, "Train End:")


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

    def __init__(self, cpu_final=False) -> None:
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
        if not isinstance(e, RuntimeError):
            return False
        if "CUDA out of memory" not in str(e):
            return False
        trainer.logger.line("cuda:{} out of memory.".format(self.idx))
        self.idx += 1
        if self.idx == len(self.devices):
            trainer.logger.line("All devices out of memory.")
            return False

        trainer.regist_device(self.devices[self.idx])
        trainer.logger.line("Change to cuda:{}".format(self.idx))
        return True

    def auto_hook(self, trainer: BaseTrainer):
        trainer.add_callback(trainer.regist_device, self)
        trainer.add_callback(trainer.train_batch, self)


class PltRecorder(TrainCallback):
    def __init__(self, monitor_vars, per_batch=5, save_epochs=(100, 1, 1), fmts=('jpg',), path=None):
        """
        :param monitor_vars:
        :param per_batch:
        :param save_epochs:
        :param fmts:
            supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
        """
        super().__init__()
        self.history = HistoryMeter()
        self.per_batch = per_batch
        self.monitor_vars = monitor_vars
        self.fmts = fmts
        assert len(save_epochs) == 3, "the length of save_epochs must equal to 3, map to (train,eval,test)"
        self.save_epochs = save_epochs
        if path is None:
            path = "plots"
        self.path = path

    def on_first_hooked(self, trainer: BaseTrainer, func, param: TrainParam):
        super().on_first_hooked(trainer, func, param)
        self.path = trainer.hold_dir(self.path)

    def _draw_history(self, val, step_key="plt_step"):
        plt.plot(self.history.get_one_history("plt_step"), self.history.get_one_history(val))
        for fmt in self.fmts:
            fn = os.path.join(self.path, "{}_{}.{}".format(val, self.history[step_key], fmt))
            plt.savefig(fn)

    def on_train_epoch_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        meter_dict = meter.var_dict()
        # self.history.merge_dict(meter_dict)
        # self.history.updata(plt_step=param.global_step)
        if param.eidx % self.save_epochs[0] == self.save_epochs[0] - 1:
            for k in self.monitor_vars:
                if k in meter_dict:
                    self._draw_history(k)
            trainer.logger.line("Curves saved in {}.(step={})".format(trainer.hold_dir(self.path), param.eidx))

    def on_train_batch_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        if param.global_step % self.per_batch == 0:
            self.history.merge_dict(meter.var_dict())
            self.history.updata(plt_step=param.global_step)

    def on_eval_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        meter_dict = meter.var_dict()
        self.history.merge_dict(meter_dict)
        self.history.updata(eval_step=param.eidx)
        if param.eidx % self.save_epochs[0] == self.save_epochs[0] - 1:
            for k in self.monitor_vars:
                if k in meter_dict:
                    self._draw_history(k, "eval_step")

    def on_test_end(self, trainer: BaseTrainer, func, param: TrainParam, meter: LogMeter, *args, **kwargs):
        meter_dict = meter.var_dict()
        self.history.merge_dict(meter_dict)
        self.history.updata(test_step=param.eidx)
        if param.eidx % self.save_epochs[0] == self.save_epochs[0] - 1:
            for k in self.monitor_vars:
                if k in meter_dict:
                    self._draw_history(k, "test_step")
