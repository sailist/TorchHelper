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
"""

import sys

sys.path.insert(0, "../")

from torchhelper.frame.callbacks import *
from torchhelper import *
from torchhelper import __version__
from torchhelper.utils.quickbuild import ToyDataLoader

print(__version__)




class MyTrainer(Trainer):

    def __init__(self, param: TrainParam):
        super().__init__(param)

        self.regist_device()  # default torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = MyModel(10)
        self.optim = optim.SGD(params=self.model.parameters(),
                               lr=0.1,
                               momentum=0.9,
                               weight_decay=0.0001,
                               nesterov=False)

        self.regist_model_and_optim(  # 用于保存模型
            model=self.model,
            optim=self.optim,
        )
        self.regist_dataloader(
            train=ToyDataLoader((10, 128), (10,)),  # train
            eval=ToyDataLoader((10, 128), (10,)),  # eval
            test=ToyDataLoader((10, 128), (10,)),  # test
        )

        self.lossf = nn.CrossEntropyLoss()

    def train_batch(self, eidx, idx, global_step, data, device, param):
        meter = LogMeter()
        meter.epoch = "({}){}/{}".format(idx, eidx, param.epoch)

        model = self.model.to(device)
        optim = self.optim

        xs, ys = data
        xs, ys = xs.to(device), ys.to(device)
        logits = model(xs)

        meter.loss = self.lossf(logits, ys)
        if eidx % 3 == 2:
            epoch_meter = self.meter(self.METER_EPOCH, HistoryMeter)
            epoch_meter.aloss = meter.loss
        optim.zero_grad()
        meter.loss.backward()
        optim.step()

        return meter

    def predict(self, xs):  # 用于测试和验证
        return self.model(xs)


if __name__ == '__main__':
    param = TrainParam()
    param.test_in_per_epoch = 1
    param.eval_in_per_epoch = 1
    # param.update_opt(
    #     epoch = 200
    # )
    param.epoch = 50
    param.build_exp_name(["epoch"])
    trainer = MyTrainer(param)

    trainer.logger.line("Any log info will be handled by Trainer.logger")

    logcall = Traininfo()
    logcall.auto_hook(trainer)

    sv = ModelCheckpoint(monitor="loss",
                         max_to_keep=3,
                         mode="min")
    sv.auto_hook(trainer)

    ec = ExpCheckpoint(10)
    ec.auto_hook(trainer)

    pr = PltRecorder(["loss"], save_epochs=(10, 1, 1))
    pr.auto_hook(trainer)

    trainer.train()

    # trainer.load_keyckpt(30)
    # trainer.train()
