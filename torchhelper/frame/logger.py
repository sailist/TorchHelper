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

    用于更快的输出日志
'''
import os
import time
from collections import Iterable, OrderedDict
from datetime import datetime
from typing import Any

from ..base.recordable import Recordable
from .parameter import LogMeter
from ..base.structure import ScreenStr


class Logger:
    def __init__(self, head: str = "", tail: str = "", asctime: bool = True, datefmt: str = '%m/%d/%Y %I:%M:%S %p',
                 sep: str = " - "):
        """
        :param head:
        :param tail:
        :param asctime:
        :param datefmt:
        :param sep:
        """
        self.head = head
        self.tail = tail
        self.asctime = asctime
        self.datefmt = datefmt
        self.out_channel = []
        self.pipe_key = set()
        self.sep = sep
        self.return_str = ""
        self.listener = []
        self.stdout = True

    def format(self, reses: Recordable, prefix="", inline=False):
        """根据初始化设置 格式化 前缀和LogMeter"""
        if self.asctime:
            cur_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S ')
        else:
            cur_date = ""

        head = "{}{}".format("\r" if inline else "", self.head)
        tail = "{}{}".format(self.tail, "" if inline else "\n")

        midstr = " ".join(
            [i for i in [prefix, self._build_logstr(reses.logdict())] if len(i) > 0])
        logstr = "".join([head, cur_date, midstr, tail])

        return logstr.rstrip("\r")

    def _build_logstr(self, ordered_dict: dict):
        return self.sep.join(["@{}={}".format(k, v) for k, v in ordered_dict.items()])

    def inline(self, meter: Recordable = None, prefix=""):
        """在一行内输出 前缀 和 LogMeter"""
        if meter is None:
            meter = LogMeter()
        logstr = self.format(meter, prefix=prefix, inline=True)
        self.handle(logstr)

    def info(self, meter: Recordable = None, prefix=""):
        """以行为单位输出 前缀 和 LogMeter"""
        if meter is None:
            meter = LogMeter()
        logstr = self.format(meter, prefix=prefix, inline=False)
        self.handle(logstr)

    def line(self,content = ""):
        """以行为单位输出文字（有时间前缀）"""
        self.info(prefix=content)

    def newline(self):
        """换行"""
        self.handle("\n")

    def handle(self, logstr, end=""):
        """
        handle log stinrg，以指定的方式输出
        :param logstr:
        :param _:
        :param end:
        :return:
        """
        for listener in self.listener:
            listener(logstr,end)

        if logstr.startswith("\r"):
            self.return_str = logstr
            self.print(ScreenStr(logstr), end=end)
        else:
            if len(self.return_str) != 0:
                self.print(self.return_str, end="\n")
            self.print(logstr,end="")

            for i in self.out_channel:
                with open(i, "a", encoding="utf-8") as w:
                    if len(self.return_str) != 0:
                        w.write("{}\n".format(self.return_str.strip()))
                    w.write(logstr)

            self.return_str = ""

    def print(self,*args,end='\n'):
        if self.stdout:
            print(*args,end=end,flush=True)

    def enbale_stdout(self,val):
        self.stdout = val

    def add_pipe(self, fn):
        """添加一个输出到文件的管道"""
        if fn in self.pipe_key:
            self.line("Add pipe {}, but already exists".format(fn))
            return False
        path, _ = os.path.split(fn)
        os.makedirs(path, exist_ok=True)
        i = 0

        fni = "{}.{}".format(fn, i)
        while os.path.exists(fni):
            i += 1
            fni = "{}.{}".format(fn, i)

        self.print("add output channel on {}".format(fni))
        self.out_channel.append(fni)
        self.pipe_key.add(fn)
        return True

    def add_log_listener(self,func):
        self.listener.append(func)
