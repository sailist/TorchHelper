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
import os
import time
from collections import OrderedDict


def get_consolo_width():
    rows, columns = os.popen('stty size', 'r').read().split()
    return int(columns)


class WalkDict(OrderedDict):
    """
    支持
    {
        "model_a"=ModelA(),
        "models_b"={
            "submodela" =SubModelA(),
            "submodelb" =SubModelB(),
        }
    }
    调用model_items()即可递归的迭代依次返回：
    "model_a"=ModelA(),
    "submodela" =SubModelA(),
    "submodelb" =SubModelB(),

    调用model_keys()可依次返回：
    ModelA(),
    SubModelA(),
    SubModelB(),
    """

    def walk_items(self):
        return self._recur_items(self)

    def walk_keys(self):
        for k, v in self.walk_items():
            yield v

    def _recur_items(self, rdict):
        for k, v in rdict.items():
            if not isinstance(v, dict):
                yield k, v
            else:
                return self._recur_items(v)


class ScreenStr(str):
    t = 0
    dt = 0.05
    offset = 0

    def __str__(self):
        return self._screen_str()

    def __repr__(self) -> str:
        return self._screen_str()

    def tostr(self):
        return super().__str__()

    @staticmethod
    def set_speed(dt: float = 0.05):
        ScreenStr.dt = dt

    def deltatime(self):
        if ScreenStr.offset == 0:
            ScreenStr.offset = time.time()
            return 0
        else:
            end = time.time()
            res = end - ScreenStr.offset
            ScreenStr.offset = end
            return res

    def cacu_offset(self, flag=False):

        if flag:
            ScreenStr.dt *= -1
        ScreenStr.t += self.deltatime() + ScreenStr.dt

        return int(ScreenStr.t)
        # return 0

    def __len__(self) -> int:
        txt = self.encode("gbk")
        return len(txt)

    def _decode_sub(self,txt,left,right):
        try:
            txt = txt[left:right].decode("gbk")
        except:
            try:
                txt = txt[left:right - 1].decode("gbk")
            except:
                try:
                    txt = txt[left + 1:right].decode("gbk")
                except:
                    txt = txt[left + 1:right-1].decode("gbk")

        return txt

    def _screen_str(self: str, margin="..."):
        width = get_consolo_width()
        txt = self.encode("gbk").strip()
        textlen = len(txt)

        if textlen <= width:
            return self

        offset = self.cacu_offset()

        left = offset
        right = width - len(margin) + offset
        if right > textlen or left < 0:
            offset = self.cacu_offset(True)
            left = offset
            right = width - len(margin) + offset

        txt = self._decode_sub(txt,left,right)

        head = "\r" if self.startswith("\r") else ""
        tail = "\n" if self.endswith("\n") else ""
        txt = "{}{}{}".format(head, txt, tail)



        return txt + margin


