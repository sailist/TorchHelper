"""
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

   人启动训练器，并且定制训练器的日志和模型保存路径，并且传递给训练器各种iter过程的参数

   训练器需要自己创造模型，创造数据集（根据参数），创造设备（根据参数）

   在训练、训练iter、验证、测试结束后，依次触发保存、日志输出等“额外”过程

   日志输出和保存需要获取的信息，应该完全能从训练器中获得，包括有哪些模型，有哪些loss或acc需要输出、有哪些超参数需要输出


"""

from .trainer import Trainer,TrainParam
from .saver import Saver
from .logger import LogMeter
from . import callbacks
from .databundler import DataBundler,ToyDataLoader