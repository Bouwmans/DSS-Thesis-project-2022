# DSS-Thesis-project-2022
Predicting future stock market movements using feedforward neural networks
This repository contains 5 code files, which just be ran sequentially

0_preprocessing processes the stock-level data, I can't publish the dataset, but the features can be recreated using the anomaly definitions file from https://www.serhiykozak.com/data

1_preprocessing_ME transforms the raw macroeconomic data from https://research.stlouisfed.org/econ/mccracken/fred-databases/ into usable features. This script also joins the stock-level and macroeconomic data into a single dataset

2_ffn_RW is used to train the different models. The hyperparameters can be easily changed from lines 230-252. After training the models the code selects the best (as measured by the evaluation metric on the validation set) configuration and epoch from the different models within that timeframe and approach. For these models predictions are made for the test set

This code uses the fastdataloader from https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6, I do not claim authorship or ownership of this code, which respectfully belongs to the original creator

3_results calculates the evaluation metrics for the testdata and constructs the portfolios

4_sharperatios calculates the Sharpe ratios for the four portfolios
