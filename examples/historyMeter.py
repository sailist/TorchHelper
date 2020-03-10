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

from torchhelper.frame.parameter import HistoryMeter

h = HistoryMeter()

h.ad += 1
h.ad = 2
h.ad = 3

h.ad += 4

h.mu *= 1
h.mu *= 2
h.mu *= 3
h.mu *= 4


h.updata(a=1,b=2)
h.updata(a=2,b=3)

print(h)
print(h.get_historys())