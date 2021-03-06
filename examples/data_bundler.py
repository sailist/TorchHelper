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

from torchhelper.frame.databundler import DataBundler
from torchhelper.utils.quickbuild import ToyDataLoader

bundler = DataBundler()
bundler.add(ToyDataLoader()).chain_iter()

xs, ys = bundler.choice_batch()

print(xs.shape, ys.shape)

x, y = bundler.choice_sample()
print(x.shape, y, y.shape)

for xs,ys in bundler:
    print(xs.shape,ys.shape)