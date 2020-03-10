'''
    Copyright (C) 2020 Shandong University

    This program is licensed under the GNU General Public License 3.0 
    (https://www.gnu.org/licenses/gpl-3.0.html). 
    Any derivative work obtained under this license must be licensed 
    under the GNU General Public License as published by the Free 
    Software Foundation, either Version 3 of the License, or (at your option) 
    any later version, if this derivative work is distributed to a third party.

    The copyright for the program is owned by Shandong University. 
    For commercial projects that require the ability to distribute 
    the code of this program as part of a program that cannot be 
    distributed under the GNU General Public License, please contact 
            
            sailist@outlook.com
             
    to purchase a commercial license.
   
    温馨提示：
        抵制不良代码，拒绝乱用代码。
        
        注意自我保护，谨防上当受骗。
        
        适当编程益脑，沉迷编程伤身。
        
        合理安排时间，享受健康生活！
'''
from functools import wraps
from typing import Any
from warnings import warn

class A:

    def __init__(self) -> None:
        super().__init__()
        print("A")


class Dynanmic:
    def __new__(cls) -> Any:
        self = super().__new__(cls)
        def wrap(func):
            @wraps(func)
            def inner(*arg, **kwargs):

                if getattr(self,"_first_called",True):
                    self._first_called = False
                    print(self._first_called)
                    self.dynamic_build(*arg,**kwargs)
                return func(*arg, **kwargs)

            return inner
        if hasattr(self,"forward"):
            self.forward = wrap(self.forward)
        else:
            warn("Dynanmic class must in the classes which have forward() method")
        return self

    def dynamic_build(self, *args, **kwargs):
        pass


class C(A, Dynanmic):
    # def forward(self, x):
    #     print("Cforward")

    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)

    def dynamic_build(self, x):
        print("firstcalled")


c = C()
c(123)
c(123)
c(123)
