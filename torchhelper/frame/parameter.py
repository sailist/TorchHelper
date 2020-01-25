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
from collections import Iterable
import fire
import torch
from torchhelper.base.recordable import Recordable

class TrainParam(Recordable):
    """用于Trainer 训练过程中获取参数使用"""
    def __init__(self):
        super().__init__(None)
        self.global_step = 0
        self.epoch = 5
        self.eidx = 0
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

    def get_exp_name(self):
        assert hasattr(self, "_exp_name"), "please first call build_exp_path()!"

        return self._exp_name

    def from_opt(self):
        def func(**kwargs):
            for k,v in kwargs.items():
                self[k] = v
        fire.Fire(func)
        return self


class LogMeter(Recordable):
    def __init__(self, default_type=None):
        super().__init__(default_type)
        self._mapfn = lambda d: d

    def set_mapfn(self, fn):
        self._mapfn = fn

    def logdict(self)->dict:
        res = self._mapfn(super().logdict())
        return res

    def updata(self,**kwargs):
        for k,v in kwargs.items():
            self[k] = v

    @property
    def k(self):
        return self._k

    def __repr__(self) -> str:
        return " - ".join(["@{}={}".format(k,v) for k,v in self.logdict().items()])
