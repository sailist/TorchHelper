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

saver = Saver("./test", max_to_keep=3)
# saver.clear_checkpoints()

for i in range(10):
    saver.checkpoint(i, {})
    print(saver._info)

saver = Saver("./test", max_to_keep=3)

for i in range(3):
    saver.checkpoint(i, {})
    print(saver._info)

for i in range(10):
    saver.checkpoint(i, {})
