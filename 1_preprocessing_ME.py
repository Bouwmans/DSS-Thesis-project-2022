# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 13:35:47 2022

@author: Koen
"""

from fredapi import Fred
import pandas as pd
import os
import numpy as np
from datetime import datetime, date
from sklearn.preprocessing import StandardScaler


path = 'D:/Koen/Msc Data Science & Society/Thesis/Data/'

pdf = None

"""The macro economic data is publicly available from: https://research.stlouisfed.org/econ/mccracken/fred-databases/
Data used in this thesis is the monthly data from the link above
"""

#csv files containig sets of macro economic variables and their transformer codes
#csvs collected in a single pandas dataframe
for csv in os.listdir("D:/Koen/Msc Data Science & Society/Thesis/Data/tabula-FRED-MD_description"):
    macros = pd.read_csv("D:/Koen/Msc Data Science & Society/Thesis/Data/tabula-FRED-MD_description/" + csv)
    try:
        pdf = pd.concat([pdf, macros])
    except:
        pdf = macros
     

def transformer(transform, data, macro):
    """
    args:
        transform (int): code for which transform to use
        data (pandas dataframe): dataframe containing macroeconomic variable
        macro (str): which variable to transform
    """
    if transform == 1:
        #no transform applied
        data[macro] = data[macro]
    elif transform == 2:
        #first order difference
        data[macro] = data[macro]- data[macro].shift(1)
    elif transform == 3:
        #second order differnece
        data[macro] = data[macro]- data[macro].shift(1)
        data[macro] = data[macro]- data[macro].shift(1)
    elif transform == 4:
        #log transform
        data[macro] = np.log(data[macro])
    elif transform == 5:
        #first order difference of log
        data[macro] = np.log(data[macro]) - np.log(data[macro].shift(1))
    elif transform == 6:
        #second order difference of log
        data[macro] = np.log(data[macro]) - np.log(data[macro].shift(1))
        data[macro] = data[macro]- data[macro].shift(1)
    elif transform == 7:
        #first order difference of percentage change
        data[macro] = (data[macro] / data[macro].shift(1)) - 1
        data[macro] = data[macro]- data[macro].shift(1)
    else:
        print("something went wrong")
    return data
    
# original dataset containing raw macro economic data
current = pd.read_csv(r"D:\Koen\Msc Data Science & Society\Thesis\Data\current.csv")

# if any variables are missing keep track in the list missing
missing = []
#loop over all the rows in pdf
for idx in range(len(pdf)):
    tmp = pdf.iloc[idx,:]
    #macro is the variable to be transformed
    macro = tmp['fred']
    #transform is the transformer code
    transform = tmp['tcode']
    #check if macro variable in the raw dataset
    if macro in current.columns:
        #select variable from the raw dataset
        data = current[['sasdate', macro]]
        #exclude first row since it does not contain data
        data = data.iloc[1:, : ]
        data = pd.DataFrame(data, columns = ['sasdate', macro])
        #transform the data
        data = transformer(transform, data, macro)
        data = data.dropna()
        data.loc[:, 'sasdate'] = pd.to_datetime(data.loc[:, 'sasdate'],infer_datetime_format=True)
        # create new dataframe of transformed data
        if idx == 0:
            macro_data = data.copy()

        else:
            data = data[macro]
            macro_data = macro_data.join(data)


    else:
        KeyError
        missing.append(macro)

start_uni = np.datetime64(datetime.date(datetime.strptime("1990-01-01", '%Y-%m-%d')))
end_uni = np.datetime64(datetime.date(datetime.strptime("2021-01-01", '%Y-%m-%d')))
end_train = np.datetime64(datetime.date(datetime.strptime("2000-01-01", '%Y-%m-%d')))

macro_data['date_ym'] = pd.to_datetime(macro_data['sasdate']).shift(2)

#mask is true for observations that lie within the investment universe, false for those outside
mask = (macro_data['date_ym'] >= start_uni) & (macro_data['date_ym'] <= end_uni)

#select only the observations that lie within the universe
macro_data = macro_data.loc[mask]

macro_data = macro_data.drop(columns = ['sasdate'])



mask_train = (macro_data['date_ym'] <= end_train)
df_train = macro_data.loc[mask_train]
macro_data['date_ym'] = macro_data['date_ym'].dt.to_period('M')
macro_data = macro_data.set_index("date_ym", drop = True)
df_train = df_train.set_index("date_ym", drop = True)

cols = df_train.columns

#initialize and fit the standard scaler on the trainingdata
ss = StandardScaler()
ss.fit(df_train.loc[:, cols])

macro_data.loc[:, cols] = ss.transform(macro_data.loc[:, cols])
macro_data = macro_data.fillna(0)
macro_data.to_csv(path + "macro_scaled_data.csv")

me = pd.read_csv(path + "macro_scaled_data.csv")
me.loc[:, 'date_ym'] = pd.to_datetime(me.loc[:, 'date_ym'],infer_datetime_format=True).dt.to_period('M')
me = me.set_index('date_ym', drop = True)
filepath_to_df = 'D:/Koen/Msc Data Science & Society/Thesis/Data/scaled_data.csv'

#join the stock-level data with the macro-economic data
df = pd.read_csv(filepath_to_df)
df.loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'],infer_datetime_format=True)
df.loc[:, 'date_ym'] = df.loc[:, 'date'].dt.to_period('M')
df = df.set_index('date_ym', drop = True)
df = df.join(me)
df = df.reset_index(drop = True)
df = df.rename_axis('idx')
df = df.reset_index(drop = False)

dates = df['date'].unique()

tmp = df[['idx', 'ret', 'date']]
tmp.to_csv(path + "/data_month/ret_file.csv", index = False)
del tmp
# df.to_csv(path + "stock_me_scaled.csv")

#save the data by month in separate csv files
for date in dates:
    tmp = df[df['date'] == date]
    tmp.to_csv(path + "/data_month/files/" + str(date)[:10] + ".csv")
