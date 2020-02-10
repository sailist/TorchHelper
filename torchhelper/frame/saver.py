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

   用于方便的保存模型和断点
'''
import json
import os
import warnings

import torch


class Saver:
    """
    用于方便的保存模型和断点，能够自动保存最近的n个模型

    Example::

        saver = Saver("./test", max_to_keep=3)
        # saver.clear_checkpoints()

        for i in range(10):
            ...
            saver.checkpoint(i, {})
    """

    def __init__(self, save_dir, ckpt_suffix="ckpth.tar", max_to_keep=1):
        assert max_to_keep >= 1, "checkpoint must larger than one."
        self._max_to_keep = max_to_keep
        self._save_dir = save_dir
        self._suffix = ckpt_suffix
        self._infofn = os.path.join(self._save_dir, "ckpt.json")
        self._info = self._initial_info()

    def _initial_info(self):
        """如果该目录是saver留下的目录，那么就加载该目录的信息，否则初始化该目录"""
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir, exist_ok=True)
            info = dict(pointers=[])
            self._updata_info(info)
            return info
        elif not os.path.exists(self._infofn):
            info = dict(
                pointers=[],
            )
            self._updata_info(info)
            return info
        else:
            def keystr2int(x):
                tryint = lambda x: int(x) if x.isnumeric() else x
                return {tryint(k): v for k, v in x.items()}

            with open(self._infofn, encoding="utf-8") as f:
                return json.load(f, object_hook=keystr2int)

    def _updata_info(self, info):
        """在save方法调用后更新维护断点信息"""
        with open(self._infofn, "w", encoding="utf-8") as w:
            json.dump(info, w)

    def _build_model_path(self, prefix, epoch: int, rewrite=False):
        """获取保存的模型路径"""
        i = 0
        path = os.path.join(self._save_dir, "{}.ckpth.tar.{}{:05d}".format(prefix, i, epoch))
        if rewrite:
            return path

        while os.path.exists(path):
            i += 1
            path = os.path.join(self._save_dir, "{}.ckpth.tar.{}{:05d}".format(prefix, i, epoch))
        return path

    def _del_pointer(self, pointer: int):
        """删除断点"""
        if pointer not in self._info:
            return

        if os.path.exists(self._info[pointer]):
            os.remove(self._info[pointer])
            os.remove("{}.json".format(self._info[pointer]))
        else:
            # TODO 仍然有一个bug，会存在一些情况下要删除的文件不存在
            warnings.warn("I don't know why checkpoint {} not exists, but it happend.")
        self._info.pop(pointer)
        self._updata_info(self._info)

    @property
    def pointers(self):
        return self._info["pointers"]

    def save(self, model: torch.nn.Module, fn_prefix: str) -> str:
        """
        保存模型，无需传入后缀，会自动的构造后缀
        :param model:
        :param fn_prefix:
        :return: 保存模型后的路径
        """
        path = os.path.join(self._save_dir, "{}.pth".format(fn_prefix))
        torch.save(model, path)
        return path

    def load(self, fn_prefix=None):
        """
        加载某个模型
        如果没有变更路径，只需要传入之前save的model即可
            会自动根据saver的路径参数和model名字构建路径
        如果要从其他的位置加载模型，那么则需要传入具体的路径path

        实际返回的是 torch.load(path) 的结果
        :param model:
        :param fn_prefix:
        :return:  torch.load(path)
        """
        path = os.path.join(self._save_dir, "{}.pth".format(fn_prefix))
        return torch.load(path)

    def _is_info(self, v):
        """判断可否被序列化为文本"""
        if isinstance(v, torch.Tensor):
            return v

        return any([isinstance(v, i) for i in {int, str, float}])

    def _format_tensor(self, v):
        """判断tensor是否可序列化（仅序列化标量）"""
        if isinstance(v, torch.Tensor):
            if len(v.shape) == 0:
                return "{:.04f}".format(v)
            else:
                return str(v.shape)
        return v

    def check_keyepoch(self, epoch, ckpt_dict):
        ckpt_dict["_saver_epoch"] = epoch
        ckpt_fn = self._build_model_path("key_model", epoch, rewrite=True)
        ckpt_info_fn = "{}.json".format(ckpt_fn)
        info_dict = {k: self._format_tensor(v) for k, v in ckpt_dict.items() if self._is_info(v)}

        with open(ckpt_info_fn, "w", encoding="utf-8") as w:
            json.dump(info_dict, w, indent=2)

        torch.save(ckpt_dict, ckpt_fn)
        return ckpt_fn

    def checkpoint(self, epoch, ckpt_dict):
        """
        保存一个断点，保存路径为建立类时传入的文件夹下的路径
        :param epoch:
        :param ckpt_dict:  与torch.save() 方法传入字典类型时的要求一样
            实际上由于torch.save() 调用时使用的是pickle，因此只要是可序列化对象都可以传入
        :return: 断点文件的路径
        """

        if len(self.pointers) == 0:
            pointer = 1
        else:
            pointer = max(self.pointers) + 1

        ckpt_dict["_saver_epoch"] = epoch
        ckpt_dict["_saver_pointer"] = pointer

        ckpt_fn = self._build_model_path("model", epoch)
        ckpt_info_fn = "{}.json".format(ckpt_fn)

        info_dict = {k: self._format_tensor(v) for k, v in ckpt_dict.items() if self._is_info(v)}

        with open(ckpt_info_fn, "w", encoding="utf-8") as w:
            json.dump(info_dict, w, indent=2)

        self._info[pointer] = ckpt_fn
        self._info["pointers"].append(pointer)
        torch.save(ckpt_dict, ckpt_fn)
        while len(self.pointers) > self._max_to_keep:
            self._del_pointer(self.pointers[0])
            self.pointers.pop(0)
        else:
            self._updata_info(self._info)

        return ckpt_fn

    def restore(self, pointer=-1):
        """
        返回一个checkpoint，实际上返回的是torch.load()的结果
        :param pointer: 返回保存的第几个断点，默认是-1，即返回最近的
        :return:
        """
        assert pointer == -1 or pointer > self._max_to_keep, "do not have this checkpoint"

        return torch.load(self._info[self.pointers[pointer]])

    def restore_keyepoch(self, epoch):
        ckpt_fn = self._build_model_path("key_model", epoch, rewrite=True)
        return torch.load(ckpt_fn)

    def append_info_to_keyepoch(self,epoch,ckpt_dict):
        ckpt_fn = self._build_model_path("key_model", epoch, rewrite=True)
        ckpt_info_fn = "{}.json".format(ckpt_fn)
        info_dict = {k: self._format_tensor(v) for k, v in ckpt_dict.items() if self._is_info(v)}

        with open(ckpt_info_fn,"r",encoding="utf-8") as r:
            old_info = json.load(r)
        for k,v in old_info.items():
            info_dict[k] = v

        with open(ckpt_info_fn, "w", encoding="utf-8") as w:
            json.dump(info_dict, w, indent=2)
        return ckpt_info_fn

    def clear_checkpoints(self):
        """删除所有的断点记录"""
        for k, v in self._info.items():  # type:(int,str)
            if isinstance(k, int):
                os.remove(v)
        self._info = dict(pointers=[])
        self._updata_info(self._info)

    def __repr__(self) -> str:
        return "[Saver,@path={} - @maxkeep={}]".format(self._save_dir, self._max_to_keep)


if __name__ == '__main__':
    saver = Saver("./test", max_to_keep=3)
    # saver.clear_checkpoints()

    for i in range(10):
        saver.checkpoint(i, {})

    for i in range(10):
        saver.checkpoint(i, {})

    for i in range(10):
        saver.checkpoint(i, {})
