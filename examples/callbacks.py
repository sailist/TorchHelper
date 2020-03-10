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
from torchhelper import TrainParam
from torchhelper.frame.callbacks import TrainCallback
from torchhelper.frame.trainer import BaseTrainer


class ACb(TrainCallback):
    priority = 0

    def on_train_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        trainer.logger.line(str(self))



class BCb(TrainCallback):
    priority = -2

    def on_train_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        trainer.logger.line(str(self))


class CCb(TrainCallback):
    priority = 2

    def on_train_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        trainer.logger.line(str(self))


class DCb(TrainCallback):
    priority = 1

    def on_train_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        trainer.logger.line(str(self))

class ECb(TrainCallback):
    priority = 1

    def on_train_begin(self, trainer: BaseTrainer, func, param: TrainParam, *args, **kwargs):
        trainer.logger.line(str(self))



from torchhelper.utils.quickbuild import ToyTrainer

trainer = ToyTrainer(TrainParam())

DCb().auto_hook(trainer)
ACb().auto_hook(trainer)
BCb().auto_hook(trainer)
CCb().auto_hook(trainer)
ECb().auto_hook(trainer)
ECb().auto_hook(trainer)

trainer.train()