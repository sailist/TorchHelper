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
from typing import Any

class Merge(type):
    """
    元类，用于将子类和父类共有字典，集合时，子类的覆盖行为改为合并父类的字典，集合

    由于用途特殊，仅识别类变量中以下划线开头的变量
    ::
        class A(metaclass=Merge):
            _dicts = {"1": 2, "3": 4}

        class B(A):
            _dicts = {"5":6,7:8}

        print(B._dicts)

    result:
    >>> {'5': 6, '3': 4, '1': 2, 7: 8}
    """

    def __new__(cls, name, bases, attrs: dict, **kwds):
        for base in bases:
            for key, value in base.__dict__.items():  # type:(str,Any)
                if not key.startswith("_"):
                    continue
                if isinstance(value, set):
                    v = attrs.setdefault(key, set())
                    v.update(value)
                elif isinstance(value, dict):
                    v = attrs.setdefault(key, dict())
                    v.update(value)

        return type.__new__(cls, name, bases, dict(attrs))

