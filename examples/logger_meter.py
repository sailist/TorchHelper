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
"""
from torchhelper.frame.parameter import LogMeter
from torchhelper.frame.logger import Logger
import time
logger = Logger()


for i in range(10000):
    for j in range(200):
        meter = LogMeter()
        meter.A_mix_mix = 0.56354
        meter.B_mix_mix = 6.56354
        meter.C_mix_mix = 4.56354
        meter.D_mix_mix = 2.56354
        logger.inline(meter,"Long LogMeter Test")
        time.sleep(1)

    logger.line("Range End")