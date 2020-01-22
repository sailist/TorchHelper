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
import torch


def classify(preds, labels, cacu_rate=False, topk=None):
    """
    用于分类的准确率
    :param preds: [batch,logits]
    :param labels: [labels,]
    :param cacu_rate: 计算正确率而不是计数
    :param topk: list(int) ，表明计算哪些topk
    :return:
        if cacu_rate:
            [topk_rate,...]
        else:
            total, [topk_count,...]
    """
    if topk is None:
        topk = (1,5)
    k = topk
    _, maxk = torch.topk(preds, max(*k), dim=-1)
    total = labels.size(0)
    test_labels = labels.view(-1, 1)  # reshape labels from [n] to [n,1] to compare [n,k]

    if cacu_rate:
        return [(test_labels == maxk[:, 0:i]).sum().item() / total for i in k]
    else:
        return total, [(test_labels == maxk[:, 0:i]).sum().item() for i in k]