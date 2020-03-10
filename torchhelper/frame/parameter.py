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
from collections import defaultdict
# from collections import Iterable
from collections.abc import Iterable
from typing import Any

import fire
import torch

from torchhelper.base.recordable import Recordable
from ..base.noneitem import NoneItem


class TrainParam(Recordable):
    """用于Trainer 训练过程中获取参数使用"""

    def __init__(self):
        super().__init__()
        self._save_in_subdir = True
        self.global_step = 0
        self.epoch = 5
        self.test_in_per_epoch = 0  # 不在训练过程中测试
        self.eval_in_per_epoch = 1
        self.eidx = 1
        self.idx = 0
        self.topk = (1, 5)
        self.auto_device = False

    def _can_in_dir_name(self, obj):
        for i in [int, float, str, bool]:
            if isinstance(obj, i):
                return True
        if isinstance(obj, torch.Tensor):
            if len(obj.shape) == 0:
                return True
        return False

    def build_exp_name(self, names: Iterable, prefix="", sep="_", ignore_mode="add"):
        prefix = prefix.strip()
        res = []
        if len(prefix) != 0:
            res.append(prefix)
        if ignore_mode == "add":
            for name in names:
                if hasattr(self, name):
                    obj = getattr(self, name)
                    if self._can_in_dir_name(obj):
                        res.append("{}={}".format(name, obj))
                else:
                    res.append(name)
        elif ignore_mode == "del":
            for name in names:
                if hasattr(self, name):
                    obj = getattr(self, name)
                    if self._can_in_dir_name(obj):
                        res.append("{}={}".format(name, obj))
        else:
            assert False

        self._exp_name = sep.join(res)
        return self._exp_name

    def set_save_in_subdir(self, val):
        self._save_in_subdir = val

    def is_save_in_subdir(self):
        return self._save_in_subdir

    def get_exp_name(self):
        assert hasattr(self, "_exp_name"), "please first call build_exp_path()!"
        return self._exp_name

    def from_opt(self):
        def func(**kwargs):
            for k, v in kwargs.items():
                self[k] = v

        fire.Fire(func)
        return self

    def update_opt(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v
        return self


class LogMeter(Recordable):
    def __init__(self):
        """
        第一次出现的变量无论什么运算符均等于和 "0" 或 "1"进行运算
        meter = LogMeter()
        meter.var += 1
        不需要事先判断该变量是否存在等逻辑
        """
        super().__init__()
        self._mapfn = lambda d: d

    def set_mapfn(self, fn):
        self._mapfn = fn

    def logdict(self) -> dict:
        res = self._mapfn(super().logdict())
        return res

    def updata(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def merge_meter(self, meter):
        # meter = meter  # type:(LogMeter)
        self._param_dict.update(meter._param_dict)
        self._format_dict.update(meter._format_dict)
        self._ignore_set.update(meter._ignore_set)
        self._short_dict.update(meter._short_dict)

    @property
    def k(self):
        return self._k

    def __repr__(self) -> str:
        return " - ".join(["@{}={}".format(k, v) for k, v in self.logdict().items()])

    def __call__(self, *args, **kwargs):
        return self._k

    def __getattr__(self, item: str):
        if item.endswith("_"):
            return item.rstrip("_")

        if item not in self._param_dict:
            return NoneItem()
        return super().__getattr__(item)

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

    def clear(self):
        self._param_dict.clear()


class HistoryMeter(LogMeter):
    def __init__(self, max=None):
        super().__init__()
        self._history = defaultdict(list)
        self._max = max

    def get_one_history(self, name, offset=0):
        return [self.to_record(i) for i in self._history[name][offset:]]

    def to_record(self, i):
        if isinstance(i, torch.Tensor):
            if len(i.shape) == 0:
                return i.item()
        elif type(i) in {int, float}:
            return i

        return None

    def get_historys(self):
        return self._history

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if not name.startswith("_"):
            self._history[name].append(value)
            while self._max is not None and len(self._history[name]) > self._max:
                self._history[name].pop(0)

    def __repr__(self) -> str:
        return " - ".join(["@{}={}".format(k, v) for k, v in self.get_historys().items()])