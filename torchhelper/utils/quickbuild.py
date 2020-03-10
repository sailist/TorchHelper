"""
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
"""
from typing import Any

from torchhelper import TrainParam
from torchhelper.frame.trainer import BaseTrainer
from torch.utils.data import DataLoader
from torch import nn
import torch



class ToyDataLoader(DataLoader):
    """用于做临时的数据提供，指定提供的数据的shape和数据batch大小即可"""

    def __init__(self, xshape=(10, 128), yshape=(10,), len=50) -> None:
        # super().__init__(None)
        self.xshape = xshape
        self.yshape = yshape
        self.len = len

    def __len__(self) -> int:
        return self.len

    def __iter__(self):
        import torch
        if self.len == -1:
            while True:
                yield torch.rand(self.xshape), torch.randint(0, 10, self.yshape, dtype=torch.long)

        for i in range(self.len):
            yield torch.rand(self.xshape), torch.randint(0, 10, self.yshape, dtype=torch.long)

class ToyModel(nn.Module):

    def __init__(self,sample_shape = (3,32,32)) -> None:
        super().__init__()

    def forward(self,xs):
        return super().forward(xs)



class ToyTrainer(BaseTrainer):


    def __init__(self, param: TrainParam):
        super().__init__(param)
        self.regist_model_and_optim()
        self.regist_dataloader(
            ToyDataLoader(),
            ToyDataLoader(),
            ToyDataLoader(),
        )

    def train_batch(self, eidx, idx, global_step, data, device, param):
        meter = self.meter(0)
        meter.loss = torch.rand(1)
        return meter

    def _test_eval_logic(self, dataloader, param: TrainParam):
        meter = self.meter(1)
        meter[1] = 10
        meter["total"] = 100
        return meter



