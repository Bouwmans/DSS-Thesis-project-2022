# -*- coding: utf-8 -*-
"""
Created on Wed May 18 20:15:59 2022

@author: Koen
"""

import pandas as pd
import numpy as np

path = "D:/Koen/Msc Data Science & Society/Thesis/"

reg = pd.read_csv(path + "RW_reg_agg_port.csv")
binary = pd.read_csv(path + "RW_bin_agg_port.csv")
mc = pd.read_csv(path + "RW_mc_agg_port.csv")
agg = pd.read_csv(path + "full_agg_portfolio.csv")

def SR(port):
    port['ls'] = port['short_ret'] + port['long_ret'] - 2
    sharpe = port.groupby('year').ls.apply(lambda x: x.mean() / x.std())
    
    port['ls'] = port['ls'] + 1
    port['cumprod'] = port['ls'].cumprod()
    return port, sharpe

reg, sreg = SR(reg)
binary, sbin = SR(binary)
mc, smc = SR(mc)
agg, sagg = SR(agg)

srs = pd.DataFrame()
srs['sr_bin'] = sbin
srs['sr_mc'] = sreg
srs['sr_reg'] = smc
srs['sr_mc'] = sagg


agg['agg_ls'] = agg['cumprod']
agg['bin_ls'] = binary['cumprod']
agg['mc_ls'] = mc['cumprod']
agg['reg_ls'] = reg['cumprod']

agg.to_csv(path + "full_agg_portfolio_2.csv")
srs.to_csv(path + "srs_all_ports.csv")
