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
import torch


def split_sub_matrix(mat: torch.Tensor, *sizes):
    """
    将一个[N,M,...,L]的矩阵按 n,m,...l 拆分成 N/n*M/m*...L/l 个 [n,m,...l]的小矩阵

    如果N/n 无法整除，不会报错而是会将多余的裁掉
    example:
        mat = torch.arange(0,24).view(4,6) # shape = [4, 6]
        >> tensor([[ 0,  1,  2,  3,  4,  5],
                   [ 6,  7,  8,  9, 10, 11],
                   [12, 13, 14, 15, 16, 17],
                   [18, 19, 20, 21, 22, 23]])

        split_sub_matrix(mat,2,3) # shape = [2, 2, 2, 3]
        >> tensor([[[[ 0,  1,  2],
                      [ 6,  7,  8]],

                     [[ 3,  4,  5],
                      [ 9, 10, 11]]],

                    [[[12, 13, 14],
                      [18, 19, 20]],

                     [[15, 16, 17],
                      [21, 22, 23]]]])

    :param mat: 一个[N,M,...L] 的矩阵
    :param sizes: n,m,...l 的list, 其长度不一定完全和mat的维数相同
        mat = torch.arange(0,240).view([4,6,10])
        split_sub_matrix(mat,2,3) # shape = [2, 2, 10, 2, 3]
    :return: 一个 [N/row,M/col,row,col] 的矩阵
    """
    for i,size in enumerate(sizes):
        mat = mat.unfold(i,size,size)
    return mat


def cartesian_product(left: torch.Tensor, right: torch.Tensor):
    """
    笛卡尔积
    example:
        cartesian_product(torch.arange(0,3),torch.arange(0,5))
        >> (tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
            tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]))
    :param mat:
    :return: 两个矩阵
    """
    # za = mat.repeat_interleave(mat.shape[0], dim=0)
    # zb = mat.repeat([mat.shape[0], 1])
    nleft = left.repeat_interleave(right.shape[0], dim=0)
    nright = right.repeat(*[item if i == 0 else 1 for i, item in enumerate(left.shape)])
    return nleft, nright


def rotation90(mat:torch.Tensor):
    return mat.transpose(2,3)

def rotation180(mat:torch.Tensor):
    return mat.flip(2)

def rotation270(mat:torch.Tensor):
    return mat.transpose(2, 3).flip(3)