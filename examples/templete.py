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
from torchhelper.frame.callbacks import Traininfo, ModelCheckpoint,DebugCallback
from torchhelper import *

class MyModel(nn.Module):
    def __init__(self,n_classes=10) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        out = x
        out = self.fc(out)
        return out

class MyTrainer(Trainer):

    def __init__(self, param: TrainParam):
        super().__init__(param)

        self.regist_device() # default torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = MyModel(10)
        self.optim = optim.SGD(params=self.model.parameters(),
                                   lr=0.1,
                                   momentum=0.9,
                                   weight_decay=0.0001,
                                   nesterov=False)

        self.regist_model_and_optim({ # 用于保存模型
            "model":self.model,
            "optim":self.optim,
        })
        self.regist_dataloader(
            train=ToyDataLoader((10, 128), (10,)),#train
            eval=ToyDataLoader((10, 128), (10,)),#eval
            test=ToyDataLoader((10, 128), (10,)),#test
        )

        self.lossf = nn.CrossEntropyLoss()

    def train_batch(self, eidx, idx, global_step, data, device, param):
        meter = LogMeter()
        meter.epoch = "({}){}/{}".format(idx,eidx,param.epoch)

        model = self.model.to(device)
        optim = self.optim

        xs,ys = data
        xs,ys = xs.to(device),ys.to(device)
        logits = model(xs)

        meter.loss = self.lossf(logits,ys)

        optim.zero_grad()
        meter.loss.backward()
        optim.step()

        return meter

    def predict(self, xs): # 用于测试和验证
        return self.model(xs)

if __name__ == '__main__':
    param = TrainParam()
    param.build_exp_name(["epoch"],prefix="mymodel")
    trainer = MyTrainer(param)

    trainer.logger.line("Any log info will be handled by Trainer.logger")

    logcall = Traininfo()
    logcall.auto_hook(trainer)

    sv = ModelCheckpoint(base_dir="./project_path", 
                         monitor="loss",
                         max_to_keep=3,
                         mode="min")
    sv.auto_hook(trainer)

    # dbg = DebugCallback()
    # dbg.reverse_hook(trainer)

    trainer.train()