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

from .parameter import LogMeter
from ..base.structure import ScreenStr


class LogParam:
    """
    meter = LogParam()
    mk = meter.k
    meter.epoch = "{}({})/{}".format(eidx, idx, self.epoch)

    ...

    meter.accuracy = cacu_acc(...)

    meter.percent(mk.accuracy) # will be format like dd.dd%

    meter.loss = loss_function(...) # will be format as dd.dddd
    meter.float(mk.loss,acc = 2) # will be format as dd.dd

    meter.loss.backward()

    meter.info("I logged some info")
    return meter  # or use 'yield meter' if in loop logits

    you can use meter.param to get the OrderedDict
    """

    class KeyObj:
        def __getattr__(self, item):
            return item

    _noset = {"param_dict", "k", "format_dict",
              "short_dict", "default_type", "_infos", "ignore_set",
              "board_dict"}

    def __init__(self, default_type=None):
        self.param_dict = OrderedDict()
        self.k = LogParam.KeyObj()
        self.default_type = default_type
        self.format_dict = {}
        self.short_dict = {}
        self.ignore_set = set()
        self.board_dict = set()
        self._infos = []  # @TODO

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in LogParam._noset:
            if name in self.board_dict:
                pass

            self.param_dict[name] = value
            if name not in self.format_dict:
                try:
                    if "{:.0f}".format(value).isdecimal():
                        self.float(name)
                except:
                    self.str(name)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        self.param_dict[key] = value

    def __getitem__(self, item):
        if item not in self.param_dict and self.default_type is not None:
            self.param_dict[item] = self.default_type()
        return self.param_dict[item]

    def merge_dict(self, obj: dict):
        for k, v in obj.items():
            self.param_dict[k] = v

    def str(self, key):
        self.format_dict[key] = "{}"

    def int(self, key):
        self.format_dict[key] = "{:.0f}"

    def short(self, key, short_key):
        self.short_dict[key] = short_key

    def ignore(self, key):
        self.ignore_set.add(key)

    def percent(self, key, acc=2):
        self.format_dict[key] = "{{:.{}%}}%".format(acc)

    def float(self, key, acc=4):
        self.format_dict[key] = "{{:.{}f}}".format(acc)

    def tensorboard(self, key):
        self.board_dict.add(key)

    @property
    def params(self):
        res = OrderedDict()
        for k, v in self.param_dict.items():
            if v is None or k in self.ignore_set:
                continue
            if k in self.format_dict:
                v = self.format_dict[k].format(v)
            name = self.short_dict.get(k, k)
            res[name] = v

        return res

    @staticmethod
    def from_dict(odict: dict):
        meter = LogParam()
        for k, v in odict.items():
            meter[k] = v
        return meter


class LogWrapper:
    """
    一个用于日志输出的装饰器。

    已废弃，但是还可以用，也还算好用，就不删除了，代码示例放在这：
    ::
        logger = Logger()

        @logger.logger(inline=True, not_in_class=True)
        def cacu(eidx):
            for i in range(10000):
                meter = LogParam()
                meter.loss = i
                meter.accr = 0.64
                meterk = meter.k
                meter.percent(meterk.accr)
                yield meter

        for i in range(100):
            cacu(i)

    `::
        @logger.logger(inline=True, not_in_class=True)
        def cacu(eidx,**kw):
            pass

        for i in range(100):
            cacu(i,p1 = "arg1")
    """

    def __init__(self, head: str = "", tail: str = "", asctime: bool = True, datefmt: str = '%m/%d/%Y %I:%M:%S %p',
                 sep: str = " - "):
        '''
        :param head:
        :param tail:
        :param asctime:
        :param datefmt:
        :param sep:
        '''

        self.head = head
        self.tail = tail
        self.asctime = asctime
        self.datefmt = datefmt
        self.out_channel = []
        self.sep = sep

    def _build_logstr(self, ordered_dict: OrderedDict):
        return self.sep.join(["@{}={}".format(k, v) for k, v in ordered_dict.items()])

    def wrap(self, func,
             inline: bool = False,
             prefix: str = "", suffix: str = "",
             cacu_time: bool = True, cacu_step=False,
             not_in_class: bool = False,
             append_fn: bool = False, append_args: bool = False, append_kws: bool = False,
             max_step=None):
        """
        使用该方法代替装饰器，
        logger = Logger()
        func = logger.wrap(func)

        该方法与下面的方法效果等同
        @logger.logger()
        def func():
            ...
        :param func:
        :param inline:
        :param prefix:
        :param suffix:
        :param cacu_time:
        :param cacu_step:
        :param not_in_class:
        :param append_fn:
        :param append_args:
        :param append_kws:
        :return:
        """

        wrapper = self.logger(inline=inline,
                              prefix=prefix, suffix=suffix,
                              cacu_time=cacu_time, cacu_step=cacu_step,
                              not_in_class=True,
                              append_fn=append_fn, append_args=append_args, append_kws=append_kws,
                              max_step=max_step)
        return wrapper(func)

    @staticmethod
    def _sec2str(s):
        """ 将秒数转化为 ddhddmdds 的格式"""
        s = int(s)
        # sec = int(s) % 60
        m = int((s - s % 60) / 60)
        h = int((m - m % 60) / 60)
        s = s % 60
        if m == 0 and h == 0:
            return "{:02d}s".format(s)
        if h == 0:
            return "{:02d}m{:02d}s".format(m, s)
        return "{:02d}h{:02d}m{:02d}s".format(h, m, s)

    def _ensure_var_exist(self, clazz, item, default):
        '''确保一个类包含某个变量 item'''
        if not hasattr(clazz, item):
            setattr(clazz, item, default)

    def logger(self,
               inline: bool = False,
               prefix: str = "", suffix: str = "",
               cacu_time: bool = True, cacu_step=False, max_step=None,
               not_in_class: bool = False,
               append_fn: bool = False, append_args: bool = False, append_kws: bool = False,
               dict_pipe=None):
        '''
        该装饰器装饰的方法，应该返回一个LogParam() 对象或者 一个字典
        如果返回一个LogParam() 对象，那么过程中存储的变量将会是有序的

        :param inline: 输出是否控制在一栏，仅在 循环逻辑中 yield meter 的时候有用
        :param prefix: 输出的前缀，注意在实例化Logger类时，还可以指定头部，头部在前缀前，一般不需要手动添加 '\r'
        :param suffix: 输出的后缀，注意在实例化Logger类时，还可以指定尾部，尾部在后缀后，一般不需要手动添加换行符
        :param cacu_time: 是否在输出中添加执行时间
        :param cacu_step: 是否在输出中添加该方法内部循环了多少次（epoch级别或batch级别）
        :param max_step: 如果是循环逻辑，那么其最大执行次数是几（多用于测试整个流程是否可以跑通）
            默认为None，即跑到正常结束
        :param not_in_class: 如果不是在类方法中使用logger装饰器，那么需要声明该方法不在类中
        :param append_fn: 是否在输出中添加执行的方法名
        :param append_args: 是否在输出中添加执行的无名称参数
        :param append_kws: 是否在输出中添加执行的有key 参数
        :param dict_pipe: 如果想对返回的meter做一个统一的处理，可以通过该方法
            该方法将接受一个LogParam对象
        :return:
        '''

        def funcwrapper(func):
            def wrapper(*args, **kw):
                start = time.time()

                reses = func(*args, **kw)
                end = time.time()

                if not not_in_class:
                    instance, *args = args

                if self.asctime:
                    cur_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S  ')
                else:
                    cur_date = ""

                head = "{}{}".format("\r" if inline else "", self.head)
                tail = "{}{}".format(self.tail, "" if inline else "\n")
                logstr = ""
                if isinstance(reses, dict):
                    meter = LogParam()
                    meter.merge_dict(reses)
                    meter.func = func.__name__ if append_fn else None
                    meter.args = args if append_args else None
                    meter.merge_dict(kw if append_kws else {"": None})
                    meter.run_time = LogWrapper._sec2str(end - start) if cacu_time else None

                    if callable(dict_pipe):
                        meter = dict_pipe(meter)

                    midstr = " ".join([i for i in [prefix, self._build_logstr(meter.params), suffix] if len(i) > 0])
                    logstr = "".join([head, cur_date, midstr, tail])

                    self.info(logstr, sep="", end="")
                elif isinstance(reses, Iterable):
                    for i, res in enumerate(reses):
                        if max_step is not None:
                            if i > max_step:
                                break

                        # if fself is not None and _cacu_step:
                        #     setattr(fself, var_name, getattr(fself, var_name) + 1)

                        end = time.time()
                        if isinstance(res, dict):
                            res = LogParam.from_dict(res)

                        if callable(dict_pipe):
                            res = dict_pipe(res)

                        if cacu_step:
                            res.step = i
                            res.int("step")

                        res.func = func.__name__ if append_fn else None
                        res.args = args if append_args else None
                        res.merge_dict(kw if append_kws else {"": None})
                        res.run_time = LogWrapper._sec2str(end - start) if cacu_time else None

                        if callable(dict_pipe):
                            res = dict_pipe(res)

                        midstr = " ".join(
                            [i for i in [prefix, self._build_logstr(res.params), suffix] if len(i) > 0])
                        logstr = "".join([head, cur_date, midstr, tail])
                        if inline:
                            self.info(logstr, sep="", end="", just_print=True)
                        # else:
                        #     self.info(logstr.strip(), sep="", end="\n", just_print=True)
                    else:
                        self.info(logstr, sep="", end="", just_out=True)
                    self.info()
                elif isinstance(reses, LogParam):
                    if callable(dict_pipe):
                        reses = dict_pipe(reses)

                    reses.func = func.__name__ if append_fn else None
                    reses.args = args if append_args else None
                    reses.merge_dict(kw if append_kws else {"": None})
                    reses.run_time = LogWrapper._sec2str(end - start) if cacu_time else None

                    midstr = " ".join(
                        [i for i in [prefix, self._build_logstr(reses.params), suffix] if len(i) > 0])
                    logstr = "".join([head, cur_date, midstr, tail])

                    self.info(logstr, sep="", end="\n")
                else:
                    midstr = self.sep.join(
                        [i for i in [prefix,
                                     self._build_logstr(OrderedDict(

                                         func=func.__name__,
                                         result=reses,
                                         args=args,
                                         **kw,
                                         run_time="{:.0f}s".format(end - start), )),
                                     suffix] if len(i) > 0])
                    logstr = "".join([head, cur_date, midstr, tail])
                    self.info(logstr, sep="", end="\n")

            return wrapper

        return funcwrapper

    def info(self, content="", sep="", end="\n", just_print=False, just_out=False):
        if not just_out:
            print(content, sep=sep, end=end, flush=True)

        if not just_print:
            for o in self.out_channel:
                with open(o, "a", encoding="utf-8") as w:
                    w.write("{}\n".format(content.strip()))

    def add_pipe(self, fn: str):
        path, _ = os.path.split(fn)
        os.makedirs(path, exist_ok=True)
        i = 0
        fni = "{}.{}".format(fn, i)
        while os.path.exists(fni):
            i += 1
            fni = "{}.{}".format(fn, i)

        print("add output channel on {}".format(fni))
        self.out_channel.append(fni)


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
        self.sep = sep
        self.return_str = ""

    def format(self, reses: LogMeter, prefix="", inline=False):
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

        return logstr

    def _build_logstr(self, ordered_dict: dict):
        return self.sep.join(["@{}={}".format(k, v) for k, v in ordered_dict.items()])

    def inline(self, meter: LogMeter = None, prefix=""):
        """在一行内输出 前缀 和 LogMeter"""
        if meter is None:
            meter = LogMeter()
        logstr = self.format(meter, prefix=prefix, inline=True)
        self.handle(logstr)

    def info(self, meter: LogMeter = None, prefix=""):
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

    def handle(self, logstr, _="", end=""):
        """
        handle log stinrg，以指定的方式输出
        :param logstr:
        :param _:
        :param end:
        :return:
        """
        if logstr.startswith("\r"):
            self.return_str = logstr
            print(ScreenStr(logstr), end=end, flush=True)
        else:
            if len(self.return_str) != 0:
                print(self.return_str, end="\n", flush=True)
            print(logstr,end="",flush=True)

            for i in self.out_channel:
                with open(i, "a", encoding="utf-8") as w:
                    if len(self.return_str) != 0:
                        w.write("{}\n".format(self.return_str.strip()))
                    w.write(logstr)

            self.return_str = ""

    def add_pipe(self, fn):
        """添加一个输出到文件的管道"""
        path, _ = os.path.split(fn)
        os.makedirs(path, exist_ok=True)
        i = 0
        fni = "{}.{}".format(fn, i)
        while os.path.exists(fni):
            i += 1
            fni = "{}.{}".format(fn, i)

        print("add output channel on {}".format(fni))
        self.out_channel.append(fni)