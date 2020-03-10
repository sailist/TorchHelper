"""
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
"""
import shutil
import time
from collections import OrderedDict


def get_consolo_width():
    return shutil.get_terminal_size().columns


def test_str(n: int = 30):
    return "\r{}".format(str([i for i in range(n)]))


class WalkDict(OrderedDict):
    """
    能够遍历字典中的嵌套字典的类

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
        """
        迭代所有的k，v，
        如果在迭代的过程中遇到了dict类，则递归迭代字典，而不是返回该字典
        """
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


class ScreenStr():
    """
    该方法用于长期输出在同一行（如batch级别的loss输出）时，控制屏幕输出位于同一行，支持中英文混合
    该方法效率比较低，需要经过一次调用系统命令，两次对文本的编码解码和（最多）三次嵌套异常处理，
    因此可能适用场景也就限于炼丹了吧（笑
    """
    t = 0
    dt = 7
    last = 0

    max_wait = 1.
    wait = 0
    wait_toggle = False

    debug = False
    last_width = 0

    def __init__(self, content="") -> None:
        self.content = content

    def __str__(self):
        return self._screen_str()

    def __repr__(self) -> str:
        return self._screen_str()

    def tostr(self):
        return self.content

    @staticmethod
    def set_speed(dt: float = 0.05):
        ScreenStr.dt = dt

    @staticmethod
    def deltatime():
        if ScreenStr.last == 0:
            ScreenStr.last = time.time()
            return 0
        else:
            end = time.time()
            res = end - ScreenStr.last
            ScreenStr.last = end
            return res

    @staticmethod
    def cacu_offset():
        delta = ScreenStr.deltatime()

        if ScreenStr.wait_toggle:
            ScreenStr.wait += abs(delta)
            if ScreenStr.wait > ScreenStr.max_wait:
                ScreenStr.wait_toggle = False
                ScreenStr.wait = 0
                ScreenStr.dt *= -1
            else:
                return int(ScreenStr.t - delta * ScreenStr.dt)

        ScreenStr.t += delta * ScreenStr.dt

        return int(ScreenStr.t)

    def __len__(self) -> int:
        txt = self.content.encode("gbk")
        return len(txt)

    def _decode_sub(self, txt, left, right):
        try:
            txt = txt[left:right].decode("gbk")
        except:
            try:
                txt = txt[left:right - 1].decode("gbk")
            except:
                try:
                    txt = txt[left + 1:right].decode("gbk")
                except:
                    txt = txt[left + 1:right - 1].decode("gbk")

        return txt

    @staticmethod
    def refresh():
        ScreenStr.t = 0
        ScreenStr.dt = abs(ScreenStr.dt)
        ScreenStr.last = 0

    @staticmethod
    def consolo_width():
        width = get_consolo_width()
        if width != ScreenStr.last_width:
            ScreenStr.refresh()
        ScreenStr.last_width = width
        return width

    @staticmethod
    def stop():
        ScreenStr.wait_toggle = True

    def _screen_str(self, margin="..."):
        width = ScreenStr.consolo_width()

        txt = self.content.encode("gbk").strip()
        textlen = len(txt)

        if textlen <= width:
            return self.content

        offset = ScreenStr.cacu_offset()

        right = width - len(margin) + offset
        if width - len(margin) + offset > textlen:
            right = textlen
            ScreenStr.stop()

        left = right + len(margin) - width
        if left < 0:
            left = 0
            ScreenStr.stop()
            right = width - len(margin)

        if self.debug:
            debug = "[o:{};l:{:.4f};r:{:.4f};]".format(offset, ScreenStr.t, ScreenStr.wait)
        else:
            debug = ""

        txt = self._decode_sub(txt, left, right - len(debug))

        head = "\r" if self.content.startswith("\r") else ""
        tail = "\n" if self.content.endswith("\n") else ""

        txt = "{}{}{}{}".format(head, debug, txt, tail)

        return txt + margin




if __name__ == '__main__':
    for i in range(100000):
        print(ScreenStr("\r{}".format(str([i for i in range(30)]))), end="", flush=True)
        time.sleep(0.01)
