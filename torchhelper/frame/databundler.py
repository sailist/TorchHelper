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

   当一个模型需要训练多个数据集的时候，通过DataBundler类和附属的装饰器灵活的切换数据集

   目标: 尽可能的减少多个数据集之间的类似的代码

   DataBundler()
'''


class DataBundler:
    def __init__(self):
        pass