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


class NoneItem:
    """
    可以和任何元素做加减乘除，在计算的过程中根据运算
    该类相当于 零元（0）或单位元（1）
    """

    @staticmethod
    def clone(x):
        pass

    def __eq__(self, o: object) -> bool:
        return o is None

    def __add__(self, other):
        return 0 + other

    def __mul__(self, other):
        return 1 * other

    def __sub__(self, other):
        return 0 - other

    def __truediv__(self, other):
        return 1 / other

    def __floordiv__(self, other):
        return other

    def __repr__(self):
        return "NoneItem()"

    def __lt__(self, other):
        return True

    def __le__(self, other):
        return True

    def __ge__(self, other):
        return True

    def __gt__(self, other):
        return True

    def __cmp__(self, other):
        return True

    def __ne__(self, other):
        return other is not None
