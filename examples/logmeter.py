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

from torchhelper import *
from torchhelper.frame.trainer import Trainer

meter = LogMeter()

meter.loss = 0.1638574968
meter.loss2 = 1.1235435465
meter.loss3 = 2.4654687987

meter.all_loss = meter.loss + meter.loss2 + meter.loss3

meter.float(meter.loss_, acc=6)
meter.ignore(meter.loss2_)
meter.short(meter.all_loss_,"AL")

print(meter.loss)
print(meter.loss_)
print(meter.k.loss)
print(meter._k.loss)

trainer = Trainer(TrainParam())
trainer.logger.info(meter, prefix="Logger prefix:")