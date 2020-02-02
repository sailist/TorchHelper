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
from collections import OrderedDict
from typing import Any

from .meta import Merge


class Recordable(metaclass=Merge):
    """
    record parameters for use and log.

    设置变量时如果前面有下划线'_'，则会被加入到类本身的变量中
    不允许变量名后包含下划线'_'，该类型变量有特殊用途
    其他变量将会被加入到param_dict中，作为超参数获取，日志输出等用途
    """
    _default_dict = dict()

    class KeyObj:
        def __getattr__(self, item):
            return item

    def __init__(self, default_type=None):
        self._param_dict = OrderedDict()
        self._k = self
        self._k = Recordable.KeyObj()
        self._default_type = default_type
        self._format_dict = {}
        self._short_dict = {}
        self._read_mode = False
        self._ignore_set = set()
        self._board_set = set()
        self._infos = []  # @TODO
        self.load_default()

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        elif name.endswith("_"):
            assert False, "attr name must not end with '_'"
        else:
            if name in self._board_set:
                pass  # TODO，添加自动调出面板？或者在callbacks中添加

            self._param_dict[name] = value
            if name not in self._format_dict and type(value) not in {int, float, bool}:
                try:
                    if "{:.0f}".format(value).isdecimal():
                        self.float(name)
                except:
                    self.str(name)
            elif name not in self._format_dict and isinstance(value, float):
                self.float(name)

    def __getattr__(self, item):
        if item not in self._param_dict:
            raise AttributeError()
        return self._param_dict[item]

    def __setitem__(self, key, value):
        self._param_dict[key] = value

    def __getitem__(self, item):
        if item not in self._param_dict and self._default_type is not None:
            self._param_dict[item] = self._default_type()
        return self._param_dict[item]

    def merge_dict(self, obj: dict):
        for k, v in obj.items():
            self._param_dict[k] = v

        return self

    def str(self, key):
        self._format_dict[key] = "{}"

    def int(self, key):
        self._format_dict[key] = "{:.0f}"

    def short(self, key, short_key):
        self._short_dict[key] = short_key

    def ignore(self, key):
        self._ignore_set.add(key)

    def percent(self, key, acc=2):
        self._format_dict[key] = "{{:.{}%}}%".format(acc)

    def float(self, key, acc=4):
        self._format_dict[key] = "{{:.{}f}}".format(acc)

    def board(self, key):
        self._board_set.add(key)

    def logdict(self):
        res = OrderedDict()
        for k, v in self._param_dict.items():
            if v is None or k in self._ignore_set:
                continue
            if k in self._format_dict:
                v = self._format_dict[k].format(v)
            name = self._short_dict.get(k, k)
            res[name] = v

        return res

    def board_dict(self):
        res = OrderedDict()
        for k, v in self._param_dict.items():
            if k not in self._board_set:
                continue
            if v is None:
                continue

            name = self._short_dict.get(k, k)
            res[name] = v
        return res

    def read_mode(self):
        self._read_mode = True

    def write_mode(self):
        self._read_mode = False

    def load_default(self):
        for k, v in self.__class__._default_dict.items():
            self.__setattr__(k, v)
