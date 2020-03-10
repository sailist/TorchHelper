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

    温馨提示：
        抵制不良代码，拒绝乱用代码。

        注意自我保护，谨防上当受骗。

        适当编程益脑，沉迷编程伤身。

        合理安排时间，享受健康生活！
"""

class AvgItem:
    """
    用于保存累积均值的类
    avg = AvgItem()
    avg += 1 # avg.update(1)
    avg += 2
    avg += 3

    avg.item = 3 #(last item)
    avg.avg = 2 #(average item)
    avg.sum = 6
    """
    def __init__(self, weight=1) -> None:
        super().__init__()
        self.sum = 0
        self.weight = weight
        self.count = 0
        self.item = None

    def __add__(self, other):
        self.update(other)

    def update(self, other, weight=1):
        self.sum += other * self.weight
        self.count += self.weight
        self.item = other

    @property
    def avg(self):
        return self.sum / self.count

    def __repr__(self) -> str:
        return str("{}({})".format(self.item, self.avg))
