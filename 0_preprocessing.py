# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 11:59:28 2022

@author: Koen
"""
import pandas as pd
import pyreadr
from datetime import datetime, date
import numpy as np 
from sklearn.preprocessing import StandardScaler
from pickle import dump
import os

path = 'D:/Koen/Msc Data Science & Society/Thesis/Data/'

result = pyreadr.read_r('D:\Koen\Msc Data Science & Society\Thesis\Data\Data_Jens\Dataset_US.rds') # also works for RData
df = result[None]


#define start and end of the investment universe, and the end of the first training set
start_uni = np.datetime64(datetime.date(datetime.strptime("1990-01-01", '%Y-%m-%d')))
end_uni = np.datetime64(datetime.date(datetime.strptime("2021-01-01", '%Y-%m-%d')))
end_train = np.datetime64(datetime.date(datetime.strptime("2000-01-01", '%Y-%m-%d')))
#set date variable to datatype date
df.loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'],infer_datetime_format=True)
# df = pdf.sort_values(by = ['date'])

#mask is true for observations that lie within the investment universe, false for those outside
mask = (df['date'] > start_uni) & (df['date'] <= end_uni)

#select only the observations that lie within the universe
df = df.loc[mask]

#count missing values, and missing values as percentage of total observations
nas = df.isna().sum()
nas_p = df.isna().sum() / len(df)

#drop indrrev and inrrevlv as more than 83% of data is missing
df = df.drop(columns = ['indrrev', 'indrrevlv'])

nas_p = df.isna().sum() / len(df)

# median_values = df.groupby('date').median()

#fill the remaining missing values by that months median
for col in df.columns:
    if col != "date" and col != 'gvkey' and col != 'permno':
        df[col] = df[col].fillna(df.groupby('date')[col].transform('median'))
        
# identifier columns, return column, date column, catagory column, and binary columns will not be standarddized
unscaled_cols = ['permno', 'gvkey', 'ret', 'date', 'exchcd', 'debtiss', 'repurch']
scaled_cols = []
for col in df.columns:
    if col not in unscaled_cols:
        scaled_cols.append(col)

#scaler only fitted on training data to prevent information leakage
mask_train = (df['date'] <= end_train)
df_train = df.loc[mask_train]

#initialize and fit the standard scaler on the trainingdata
ss = StandardScaler()
ss.fit(df_train.loc[:, scaled_cols])

#create folder for saving the standard scaler
try:
    os.mkdir(path + "/standardscaler/")
except:
    FileExistsError

#save standard scaler for later use
dump(ss, open(path + "/standardscaler/std_scaler.pkl", "wb"))
#ss = load(open(path + "/standardscaler/std_scaler.pkl", 'rb'))


#transform the full dataset based on the mean and stdev of the trainingdata
df.loc[:, scaled_cols] = ss.transform(df.loc[:, scaled_cols])

#binary value that equals one if ret is larger or equal to zero, zero otherwise
df.loc[:, "ret_binary"] = np.where(df.ret >= 0, 1, 0)

mc_labels = []
for i in range(10):
    mc_labels.append(i)

df['ret_multiclass'] = df.groupby('date').ret.transform(lambda x: pd.qcut(x, q = 10, labels = mc_labels).astype(str))


#save the scaled dataset to a csv_file
df.to_csv(path + "scaled_data.csv", index = False)

# df = pd.read_csv(path + "scaled_data.csv")