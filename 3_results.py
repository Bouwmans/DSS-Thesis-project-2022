# -*- coding: utf-8 -*-
"""
Created on Mon May 16 10:11:20 2022

@author: Koen
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix

def cumprod(port):
    port = pd.DataFrame(port.groupby('date').ret.mean() + 1)
    port['cum_prod'] = port.cumprod()
    return port

filepath_to_dfs = 'D:/Koen/Msc Data Science & Society/Thesis/Data/data_month/files/'
files = os.listdir(filepath_to_dfs)

# pdf = None
# for file in files:
#     if pdf is not None:
#         tmp = pd.read_csv(filepath_to_dfs + file, usecols=["date", "permno"])
#         pdf = pd.concat([pdf, tmp])
#     elif pdf is None:
#         pdf = pd.read_csv(filepath_to_dfs + file, usecols=["date", "permno"])
        
# print(len(pdf['permno'].unique()))
# print(np.mean(pdf.groupby('date').count()))

# pdf.loc[:, 'year'] = pdf.loc[:, 'date'].transform(lambda x: x[:4])

# counts_y = pdf.groupby('year').permno.nunique()

# counts_y.to_csv('D:/Koen/Msc Data Science & Society/Thesis/year_counts.csv')

filepath_to_results = 'D:/Koen/Msc Data Science & Society/Thesis/results/'

aps = os.listdir(filepath_to_results)
for ap in aps:
    mc_labels = []
    for i in range(10):
        mc_labels.append(i)
    if ap == "RW_bin":
        predictions = pd.read_csv(filepath_to_results + ap + "/0/predictions.csv", 
                                  usecols=['date', 'permno','ret_binary','ret'])
        full_agg = predictions[['date', 'permno', 'ret']].copy()
        # full_agg = full_agg.set_index(keys = ['date', 'permno'])
        predictions = predictions.set_index(keys = ['date', 'permno'])
        
        
        accs = pd.read_csv(filepath_to_results + ap + "/0/acc_year.csv", usecols=['year', 'ret_binary'])
        accs = accs.set_index('year', drop = True)
        for mod in os.listdir(filepath_to_results + ap):
            preds = pd.read_csv(filepath_to_results + ap +  "/" + mod + "/predictions.csv", 
                                      usecols=['date', 'permno', 'predictions'])
            preds = preds.set_index(keys = ['date', 'permno'])
            preds = preds.rename(columns = {'predictions': "preds_" + mod})
            predictions = predictions.join(preds, how = 'left')
            acc = pd.read_csv(filepath_to_results + ap + "/" + mod + "/acc_year.csv", usecols=['year', 'right_class'])
            acc = acc.set_index('year', drop = True)
            acc = acc.rename(columns = {'right_class': "acc_model_" + mod})
            accs = accs.join(acc, how = 'left')
        tmp = predictions.T
        tmp = tmp.drop(labels = ['ret', 'ret_binary'])
        means = tmp.mean()
        predictions['mean_predictions'] = means
        predictions = predictions.reset_index()

        predictions['year'] = np.array(predictions['date']).astype('datetime64[Y]').astype(int) + 1970
        predictions['binary_pred'] = np.where(predictions.mean_predictions < 0.5 ,0, 1)
        predictions['right_class'] = np.where(predictions.ret_binary == predictions.binary_pred,1, 0)
        cm = confusion_matrix(predictions['ret_binary'], predictions['binary_pred'])
        cm.to_csv('D:/Koen/Msc Data Science & Society/Thesis/bin_confusion_matrix.csv')
        acc_year = predictions.groupby('year')[['right_class', 'ret_binary']].mean()
        acc_year.to_csv('D:/Koen/Msc Data Science & Society/Thesis/binary_aggregate_acc.csv')
        testset = predictions[['date', 'year', 'ret', 'mean_predictions', 'permno']]
        testset['ret_group'] = testset.groupby('date').mean_predictions.transform(lambda x: pd.qcut(x.rank(method='first'), q = 10, labels = mc_labels)).astype(int)
        
        full_agg['ret_group_bin'] = testset['ret_group']
        
        short = testset[testset['ret_group'] == 0]
        long = testset[testset['ret_group'] == 9]
        
        

        accs.to_csv('D:/Koen/Msc Data Science & Society/Thesis/binary_acc.csv')
        
    
    
    if ap == "RW_mc":
        # predictions = pd.read_csv(filepath_to_results + ap + "/0/predictions.csv", 
        #                           usecols=['date', 'permno','ret_multiclass','ret'])
        del predictions
        accs = pd.read_csv(filepath_to_results + ap + "/0/acc_year.csv", usecols=['year'])
        accs = accs.set_index('year', drop = True)
        accs.loc[:, "benchmark"] = 0.1
        for mod in os.listdir(filepath_to_results + ap):
            try:
                preds = pd.read_csv(filepath_to_results + ap +  "/" + mod + "/predictions.csv")
                predictions = pd.concat([predictions, preds])
            except NameError:
                predictions = pd.read_csv(filepath_to_results + ap +  "/" + mod + "/predictions.csv")
            acc = pd.read_csv(filepath_to_results + ap + "/" + mod + "/acc_year.csv", usecols=['year', 'right_class'])
            acc = acc.set_index('year', drop = True)
            acc = acc.rename(columns = {'right_class': "acc_model_" + mod})
            accs = accs.join(acc, how = 'left')
        
        means = predictions.groupby(['date', 'permno']).mean()
        tmp = means.drop(labels = ['ret', 'ret_multiclass', 'year'], axis = 1)
        tmp = tmp.T
        classes = tmp.idxmax()
        
        predictions = pd.read_csv(filepath_to_results + ap + "/0/predictions.csv", 
                                  usecols=['date', 'permno','ret_multiclass','ret', 'year'])
        
        predictions = predictions.set_index(keys = ['date', 'permno'])
        predictions['predicted_classes'] = classes
        print(predictions.dtypes)
        predictions = predictions.astype({'predicted_classes': 'int64'})
        predictions = predictions.reset_index()
        predictions['right_class'] = np.where(
            predictions.ret_multiclass == predictions.predicted_classes, 1, 0)
    
        cm = confusion_matrix(predictions['ret_multiclass'], predictions['predicted_classes'])
        cm.to_csv('D:/Koen/Msc Data Science & Society/Thesis/mc_confusion_matrix.csv')

        
        acc_year = predictions.groupby('year').right_class.mean()
        acc_year.to_csv('D:/Koen/Msc Data Science & Society/Thesis/mc_aggregate_acc.csv')
        predictions['ret_group'] = predictions.groupby('date').predicted_classes.transform(lambda x: pd.qcut(x.rank(method='first'), q = 10, labels = mc_labels)).astype(int)
        # print(predictions)
        
        full_agg['ret_group_mc'] = predictions['ret_group']
        # print(full_agg)

        short = predictions[predictions['ret_group'] == 0]
        long = predictions[predictions['ret_group'] == 9]



        accs.to_csv('D:/Koen/Msc Data Science & Society/Thesis/mc_acc.csv')
        tmp = predictions.T
        tmp = tmp.drop(labels = ['ret', 'ret_multiclass'])
    
    if ap == "RW_reg":
        predictions = pd.read_csv(filepath_to_results + ap + "/0/predictions.csv", 
                                  usecols=['date', 'permno','ret'])
        predictions = predictions.set_index(keys = ['date', 'permno'])
        
        accs = pd.read_csv(filepath_to_results + ap + "/0/r2.csv", usecols=['year'])
        accs = accs.set_index('year', drop = True)
        accs.loc[:, "benchmark"] = 0.0
        for mod in os.listdir(filepath_to_results + ap):
            preds = pd.read_csv(filepath_to_results + ap +  "/" + mod + "/predictions.csv", 
                                      usecols=['date', 'permno', 'predictions'])
            preds = preds.set_index(keys = ['date', 'permno'])
            preds = preds.rename(columns = {'predictions': "preds_" + mod})
            predictions = predictions.join(preds, how = 'left')
        
            
            acc = pd.read_csv(filepath_to_results + ap + "/" + mod + "/r2.csv", usecols=['year', '0'])
            acc = acc.set_index('year', drop = True)
            acc = acc.rename(columns = {'0': "r_2_" + mod})
            accs = accs.join(acc, how = 'left')
        accs.to_csv('D:/Koen/Msc Data Science & Society/Thesis/r2_acc.csv')
        tmp = predictions.T
        tmp = tmp.drop('ret')
        means = tmp.mean()
        predictions['mean_predictions'] = means
        predictions = predictions.reset_index()

        predictions['sq_error'] = (predictions['mean_predictions'] - predictions['ret']) ** 2
        predictions['year'] = np.array(predictions['date']).astype('datetime64[Y]').astype(int) + 1970
        
        year_means = predictions.groupby('year').ret.mean()
        predictions = predictions.join(year_means, on = 'year', rsuffix = '_mean')
        predictions['ret_yvar'] = (predictions['ret'] - 0) ** 2
        r_2_years = pd.DataFrame(1 - (predictions.groupby('year').sq_error.sum() / predictions.groupby('year').ret_yvar.sum()))
        
        r_2_years.to_csv('D:/Koen/Msc Data Science & Society/Thesis/r2_aggregate_acc.csv')
        
        testset = predictions[['date', 'year', 'ret', 'mean_predictions']]
        testset['ret_group'] = testset.groupby('date').mean_predictions.transform(lambda x: pd.qcut(x.rank(method='first'), q = 10, labels = mc_labels)).astype(int)
        
        # full_agg['ret_group_reg'] = testset['ret_group']
        
        short = testset[testset['ret_group'] == 0]
        long = testset[testset['ret_group'] == 9]
        
    short['ret'] = short['ret'] * -1
    short = cumprod(short)
    long = cumprod(long)
    short['short_port'] = short['cum_prod']
    short['short_ret'] = short['ret']
    long['long_port'] = long['cum_prod']
    long['long_ret'] = long['ret']
    short = short.drop(columns = ['cum_prod', 'ret'], errors = 'ignore')
    long = long.drop(columns = ['cum_prod', 'ret'], errors = 'ignore')
    
    portfolio = short.join(long)
    portfolio = portfolio.reset_index()
    portfolio['year'] = np.array(portfolio['date']).astype('datetime64[Y]').astype(int) + 1970
    portfolio = portfolio[portfolio['year'] < 2020]
    
    portfolio.to_csv("D:/Koen/Msc Data Science & Society/Thesis/" + ap + "_agg_port.csv")
    
    
    full_agg['rank_sum'] = full_agg['ret_group_bin'] + full_agg['ret_group_mc'] + full_agg['ret_group_reg']
    
    full_agg['agg_rank'] = full_agg.groupby('date').rank_sum.transform(lambda x: pd.qcut(x.rank(method='first'), q = 10, labels = mc_labels)).astype(int)
    short = full_agg[full_agg['agg_rank'] == 0]
    long = full_agg[full_agg['agg_rank'] == 9]
    
    short['ret'] = short['ret'] * -1
    short = cumprod(short)
    long = cumprod(long)
    short['short_port'] = short['cum_prod']
    short['short_ret'] = short['ret']
    long['long_port'] = long['cum_prod']
    long['long_ret'] = long['ret']
    short = short.drop(columns = ['cum_prod', 'ret'], errors = 'ignore')
    long = long.drop(columns = ['cum_prod', 'ret'], errors = 'ignore')
    
    portfolio = short.join(long)
    portfolio = portfolio.reset_index()
    portfolio['year'] = np.array(portfolio['date']).astype('datetime64[Y]').astype(int) + 1970
    # portfolio = portfolio[portfolio['year'] < 2020]
    
    portfolio.to_csv("D:/Koen/Msc Data Science & Society/Thesis/full_agg_portfolio.csv")