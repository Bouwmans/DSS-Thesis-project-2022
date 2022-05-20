# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:45:41 2022

@author: Koen
"""

import shutil
from datetime import datetime
import torch
from torch import nn, optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
# from sklearn.metrics import r2_score
# from torchsummary import summary
import os
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

path = 'D:/Koen/Msc Data Science & Society/Thesis/models/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print("Device: {}".format(device))

#setting random seed for reproducability
torch.manual_seed(3)



def file_set(start, end, files):
    file_list = []
    for file in files:
        if file > start and file <= end:
            file = str(file)[:10] + ".csv"
            file_list.append(file)
    return file_list


def pdf_builder(files, filepath_to_dfs, label, feature_set, me_set, istest = None):
    pdf = None
    for file in files:
        if pdf is not None:
            tmp = pd.read_csv(filepath_to_dfs + file)
            pdf = pd.concat([pdf, tmp])
        elif pdf is None:
            pdf = pd.read_csv(filepath_to_dfs + file)
    pdf = pdf.reset_index(drop = True)
    y = pdf[label]
    if label != 'ret':
        test_set = pdf[['date', label, 'ret', 'permno']]
    else:
        test_set = pdf[['date', label, 'permno']]
    test_set['date'] = pd.to_datetime(test_set.loc[:, 'date'],infer_datetime_format=True)
    pdf = pdf[feature_set + me_set]
    pdf = np.array(pdf).astype(np.float32)
    pdf = torch.tensor(pdf)

    y = y.astype(np.float32)
    y = torch.tensor(y)
    if istest == None:
        return pdf, y
    else:
        return pdf, y, test_set

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
    
    
class Model(nn.Module):
    def __init__(self, dropout, feature_set, hidden_layers, me_set):
        """
        Pytorch neural network, __init__ defines the network, forward is the forward pass
        Args:
            dropout (float): percentages of neurons to be randomly dropout during training
            features_set (list): feature set used for prediction, used to determine the
            amount of in_features (input features)
            hidden layers (list): a list containing tuples, the first element of each tuple
            is the number of neurons in each layer, the second element of the tuple is the
            activation function
        """
        super(Model, self).__init__()
        self.layers = nn.ModuleList()
        self.n_inputs = len(feature_set) + len(me_set)
        self.num_layers = len(hidden_layers)
        m = 0
        for size, activation, batchnorm in hidden_layers:
            m += 1
            self.layers.append(nn.Linear(self.n_inputs, size))
            self.n_inputs = size
            if batchnorm is not None:
                self.batchnorm = batchnorm
            if activation is not None:
                self.layers.append(activation)
            if m < self.num_layers:
                self.layers.append(nn.Dropout(dropout))
                
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
    

def new_date(init_date, i):
    date = (init_date.astype('M8[Y]') + np.timedelta64(i,'Y')).astype('M8[D]')
    return date

def construct_layers(n_neurons, batch_norm, label):
    hidden_layers = []
    for n in n_neurons:
        if batch_norm == True:
            hidden_layers.append((n, nn.ReLU(), nn.BatchNorm1d(n)))
        else:
            hidden_layers.append((n, nn.ReLU(), None))
    if label == "ret":
        hidden_layers.append((1, None, None))
    elif label == "ret_binary":
        hidden_layers.append((1, nn.Sigmoid(), None))
    elif label == "ret_multiclass":
        loss = hidden_layers.append((10, nn.Softmax(-1), None))   
    
    return hidden_layers

def cumprod(port):
    port = pd.DataFrame(port.groupby('date').ret.mean() + 1)
    port['cum_prod'] = port.cumprod()
    return port

"""
The date variables below define the start and ending of the training/validation/testing set
start_val is also the end of the trainingset
start_test is also the end of the validation set
"""

init_start_train = np.datetime64(datetime.date(datetime.strptime("1990-01-01", '%Y-%m-%d')))
init_start_val = np.datetime64(datetime.date(datetime.strptime("2005-01-01", '%Y-%m-%d')))
init_start_test = np.datetime64(datetime.date(datetime.strptime("2010-01-01", '%Y-%m-%d')))
init_end_test = np.datetime64(datetime.date(datetime.strptime("2011-01-01", '%Y-%m-%d')))


"""set parameters for dataset class
   Filepath points to the location of the csv dataset
   label is the dependent variable
   feature_set is a list of all features used for prediction"""
   
filepath_to_label = 'D:/Koen/Msc Data Science & Society/Thesis/Data/data_month/ret_file.csv'
filepath_to_dfs = 'D:/Koen/Msc Data Science & Society/Thesis/Data/data_month/files/'
label = 'ret'
feature_set = ['size', 'value', 'prof',
            'valprof', 'F_score', 'debtiss', 'repurch', 'nissa', 'accruals',
            'growth', 'aturnover', 'gmargins', 'ep', 'cfp', 'noa', 'inv', 'invcap',
            'igrowth', 'sgrowth', 'lev', 'roaa', 'roea', 'sp', 'gltnoa', 'mom',
            'indmom', 'valmom', 'valmomprof', 'mom12', 'momrev', 'lrrev', 'valuem',
            'nissm', 'strev', 'ivol', 'beta', 'season', 'indmomrev', 'price', 'age', 'shvol']

me_set = ['RPI', 'W875RX1', 'INDPRO', 'IPFPNSS', 'IPFINAL', 'IPCONGD', 'IPDCONGD',
       'IPNCONGD', 'IPBUSEQ', 'IPMAT', 'UEMPMEAN', 'UEMPLT5', 'UEMP5TO14', 'UEMP15OV', 
       'UEMP15T26', 'UEMP27OV', 'CLAIMSx', 'PAYEMS', 'USGOOD', 'CES1021000001',
       'USCONS', 'MANEMP', 'DMANEMP', 'NDMANEMP', 'SRVPRD', 'USTPU',
       'USWTRADE', 'USTRADE', 'USFIRE', 'USGOVT', 'CES0600000007', 'AWOTMAN', 
       'AWHMAN', 'CES0600000008', 'CES2000000008', 'CES3000000008', 'HOUST', 
       'HOUSTNE', 'HOUSTMW', 'HOUSTS', 'HOUSTW', 'PERMIT', 'PERMITNE', 'PERMITMW', 
       'PERMITS', 'PERMITW', 'DPCERA3M086SBEA', 'CMRMTSPLx', 'RETAILx', 'ACOGNO',
       'AMDMNOx', 'ANDENOx', 'AMDMUOx', 'BUSINVx', 'ISRATIOx', 'UMCSENTx',
        'M1SL', 'M2SL', 'M2REAL', 'TOTRESNS', 'NONBORRES', 'BUSLOANS', 
        'REALLN', 'NONREVSL', 'CONSPI', 'DTCOLNVHFNM', 'DTCTHFNM', 'INVEST', 
        'FEDFUNDS', 'CP3Mx', 'TB3MS', 'TB6MS', 'GS1', 'GS5', 'GS10', 'AAA', 'BAA', 
        'COMPAPFFx', 'TB3SMFFM', 'TB6SMFFM', 'T1YFFM', 'T5YFFM', 'T10YFFM', 'AAAFFM', 
        'BAAFFM', 'EXSZUSx', 'EXJPUSx', 'EXUSUKx', 'EXCAUSx', 'WPSFD49207',
        'WPSFD49502', 'WPSID61', 'WPSID62', 'OILPRICEx', 'PPICMM', 'CPIAUCSL',
        'CPIAPPSL', 'CPITRNSL', 'CPIMEDSL', 'CUSR0000SAC', 'CUSR0000SAD', 'CUSR0000SAS', 
        'CPIULFSL', 'CUSR0000SA0L2', 'CUSR0000SA0L5', 'PCEPI', 'DDURRG3M086SBEA', 'DNDGRG3M086SBEA',
        'DSERRG3M086SBEA', 'S&P 500', 'S&P: indust', 'S&P div yield', 'S&P PE ratio', 'IPDMAT','IPNMAT',
         'IPMANSICS','IPFUELS','CUMFNS','HWI','HWIURATIO','CLF16OV','CE16OV','UNRATE']



"""
path_to_models is the folder where the models are stored
num_epochs are the number of epochs for which the model trains
dropout is the percentage of neurons that are set to zero in each layer during training
hidden_layers is a list of tuples, each tuple contains the number of neurons in a layer followed by
its activation function. Activation function for the final layer is None as the output can take
any value
"""

path_to_models = "D:/Koen/Msc Data Science & Society/Thesis/models_RW/"
num_epochs = 100


try: 
    os.mkdir(path_to_models)
except FileExistsError:
    a = "b"

batch_sizes = [1024]
list_of_n_neurons = [
    [64, 32, 16],
    [512, 256, 128, 64, 32]
    ]

batch_norms = [True, False] #

dropouts = [0, 0.5]

learning_rates = [0.001]

loss_functions = [nn.MSELoss(reduction='mean')]

weigth_decays = [0]



def train_RW(i, init_start_train, init_start_val, init_start_test, init_end_test, batch_norms,
             path_to_models, filepath_to_dfs, num_epochs, batch_sizes, list_of_n_neurons,
             dropouts, learning_rates, loss_functions, weigth_decays, label, 
             me_set, feature_set, approach):
    
    start_train = init_start_train
    start_val = new_date(init_start_val, i)
    start_test = new_date(init_start_test, i)
    end_test = new_date(init_end_test, i)
    if i > 4:
        start_train = new_date(init_start_train, (i-5))
    rw_path = path_to_models + "RW_" + approach + "/" + str(i)
    try:
        os.makedirs(rw_path)
    except FileExistsError:
        a= 'b'
    

    
    files = os.listdir(filepath_to_dfs)
    files = pd.Series([file[:-4] for file in files])
    files = list(pd.to_datetime(files,infer_datetime_format=True))
    
    print("start building file lists")
    files_train = file_set(start_train, start_val, files)
    files_val = file_set(start_val, start_test, files)
    files_test = file_set(start_test, end_test, files)
    
    X_train, y_train = pdf_builder(files_train, filepath_to_dfs, label, feature_set, me_set)
    X_val, y_val = pdf_builder(files_train, filepath_to_dfs, label, feature_set, me_set)

    print("Done building files")
    for batch_size in batch_sizes:    
        train_loader = FastTensorDataLoader(X_train, y_train, batch_size = batch_size, shuffle = True)
        val_loader = FastTensorDataLoader(X_val, y_val, batch_size = batch_size, shuffle = True)
        for batch_norm in batch_norms:
            for n_neurons in list_of_n_neurons:
                hidden_layers = construct_layers(n_neurons, batch_norm, label)
                for dropout in dropouts:
                    for lr in learning_rates:
                        for decay in weigth_decays:
                            
                            model = Model(dropout, feature_set, hidden_layers, me_set)
                            model = model.to(device)
                            optimizer = optim.Adam(model.parameters(), lr, weight_decay = decay)

                            if label == "ret":
                                loss = nn.MSELoss(reduction = 'mean')
                            elif label == "ret_binary":
                                loss = nn.BCELoss(reduction = 'mean')
                            elif label == "ret_multiclass":
                                loss = nn.CrossEntropyLoss(reduction = 'mean')   
                        
                            models = os.listdir(rw_path)
                            if len(models) < 1:
                                model_num = "model_0"
                            else:
                                model_num = 0
                                models = [i.split('_', 1)[1] for i in models]
                                for mod in models:
                                    if int(mod) > model_num:
                                        model_num = int(mod)
                                model_num += 1
                                model_num = "model_" + str(model_num)
                                
                            os.mkdir(rw_path + "/" + model_num)
                            os.mkdir(rw_path + "/" + model_num + "/model_backups")
                            #config contains information about the network
                            config = pd.DataFrame(data = [[model_num, start_train, start_val, start_test, 
                                                           dropout, n_neurons, lr, str(optimizer), 
                                                           batch_size, batch_norm, str(loss)]], 
                                                  columns = ["model_num", "start_train", "start_val",
                                                             "end_val", "dropout", "network_size", 
                                                             "learning_rate", "optimizer", "batch_size",
                                                             "batch_norm", "loss"])
                            
                            config.to_csv(rw_path + "/" + model_num + "/config.csv", index = False)
                            
                            loss_df = pd.DataFrame(columns = ["epoch", "train_loss", "val_loss"])
            
                            for epoch in range(num_epochs):
                                ep_loss = 0
                                model.train()
                                
                                for features, ret in train_loader:
                                    features = features.to(device)
                                    if approach == "mc":
                                        ret = ret.type(torch.LongTensor).view(-1)
                                    ret = ret.to(device)
                                
                                    optimizer.zero_grad()
                                    predictions = model(features)
                                    if approach != "mc":
                                        predictions = model(features).reshape(-1)
                                    
                                    del features
                                    batch_loss = loss(predictions, ret)
                                    batch_loss.backward()
                                    optimizer.step()
                                    ep_loss += float(batch_loss)
                                    del batch_loss
                                ep_loss = ep_loss / len(train_loader)
                                val_loss = 0
                                model.eval()
                                for features, ret in val_loader:
                                    features = features.to(device)
                                    if approach == "mc":
                                        ret = ret.type(torch.LongTensor).view(-1)

                                    ret = ret.to(device)
                                    predictions = model(features)
                                    if approach != "mc":
                                        predictions = model(features).reshape(-1)
                                    del features
                                    batch_loss = loss(predictions, ret)
                                    val_loss += float(batch_loss)
                                    del batch_loss
                                val_loss = val_loss / len(val_loader)

                                
                                ep_df = pd.DataFrame(data = [[epoch, ep_loss, val_loss]], 
                                                     columns = ["epoch", "train_loss", "val_loss"])
                                loss_df = pd.concat([loss_df, ep_df])
                                torch.save(model.state_dict(), rw_path + "/" + model_num + "/model_backups/epoch_" + str(epoch))
                                loss_df.to_csv(rw_path + "/" + model_num + "/loss_history.csv", index = False)
                            del model
                            del loss
                            del optimizer


    
label = "ret"
approach = "reg"
    
for i in range(10):
    train_RW(i, init_start_train, init_start_val, init_start_test, init_end_test, batch_norms,
              path_to_models, filepath_to_dfs, num_epochs, batch_sizes, list_of_n_neurons,
              dropouts, learning_rates, loss_functions, weigth_decays, label, 
              me_set, feature_set, approach)
    
label = "ret_binary"
approach = "bin"

for i in range(10):
    train_RW(i, init_start_train, init_start_val, init_start_test, init_end_test, batch_norms,
              path_to_models, filepath_to_dfs, num_epochs, batch_sizes, list_of_n_neurons,
              dropouts, learning_rates, loss_functions, weigth_decays, label, 
              me_set, feature_set, approach)
    
label = "ret_multiclass"
approach = "mc"

for i in range(10):
    train_RW(i, init_start_train, init_start_val, init_start_test, init_end_test, batch_norms,
              path_to_models, filepath_to_dfs, num_epochs, batch_sizes, list_of_n_neurons,
              dropouts, learning_rates, loss_functions, weigth_decays, label, 
              me_set, feature_set, approach)


path_to_thesis = r"D:\Koen\Msc Data Science & Society\Thesis"
path_to_aps = r"D:\Koen\Msc Data Science & Society\Thesis\models_RW"
aps = os.listdir(path_to_aps)

try:
    os.mkdir(path_to_thesis + "/results/")
    for ap in aps:
        os.mkdir(path_to_thesis + "/results/" + ap)
except:
    FileExistsError

for ap in aps:
    if ap == 'RW_bin':
        label = "ret_binary"
        approach = "bin"
    elif ap == 'RW_mc':
        label = "ret_multiclass"
        approach = "mc"
    elif ap == "RW_reg":
        label = "ret"
        approach = "reg"
    years = os.listdir(path_to_aps + "/" + ap)
    for year in years:
        models = os.listdir(path_to_aps + "/" + ap + "/" + year)
        errors = pd.DataFrame(columns = ['model_no', 'epoch', 'train_loss', 'val_loss'])
        
        path_to_year = path_to_thesis + "/results/" + ap + "/" + str(year)
        
        try:
            os.mkdir(path_to_year) 
        except:
            FileExistsError
            
        for model in models:
            try:
                tmp = pd.read_csv(path_to_aps + "/" + ap + "/" + year + "/" + model + "/loss_history.csv", 
                                  usecols = ['epoch', 'train_loss', 'val_loss'])
                tmp['model_no'] = model
                errors = pd.concat([errors, tmp])
            except:
                FileNotFoundError
        errors = errors.reset_index(drop = True)
        idx_min = errors['val_loss'].idxmin()
        # best_year_model = errors.iloc[idx_min, :]
        model = errors.loc[idx_min, 'model_no']
        epoch = errors.loc[idx_min, 'epoch']
        
        
        try:
            config = pd.read_csv(path_to_aps + "/" + ap + "/" + year + "/" + model + "/config.csv")
            config['epoch'] = epoch
            config['year'] = year
            configs = pd.concat([configs, config])
        except NameError:
            configs = pd.read_csv(path_to_aps + "/" + ap + "/" + year + "/" + model + "/config.csv")
            configs['epoch'] = epoch
        
        config = pd.read_csv(path_to_aps + "/" + ap + "/" + year + "/" + model + "/config.csv")
        dropout = config.loc[0, 'dropout']
        layers = config.loc[0, 'network_size']
        layers = layers[1:-1].split(", ")
        n_neurons = []
        for layer in layers:
            n_neurons.append(int(layer))
        
        start_test = config.loc[0, 'end_val']
        
        batch_norm = config.loc[0, 'batch_norm']
        hidden_layers = construct_layers(n_neurons, batch_norm, label)
        model_ = Model(dropout, feature_set, hidden_layers, me_set)
        path_to_mod = path_to_aps + "/" + ap + "/" + year + "/" + model + "/model_backups/epoch_" + str(epoch)
        model_.load_state_dict(torch.load(path_to_mod))
        init_start_test = np.datetime64(datetime.date(datetime.strptime(start_test, '%Y-%m-%d')))
        start_test_year =  init_start_test.astype('datetime64[Y]').astype(int) + 1970
        init_end_test = np.datetime64(datetime.date(datetime.strptime("2021-01-01", '%Y-%m-%d')))
        # print(model_)
        model_.eval()
        files = os.listdir(filepath_to_dfs)
        files = pd.Series([file[:-4] for file in files])
        files = list(pd.to_datetime(files,infer_datetime_format=True))
        files_test = file_set(init_start_test, init_end_test, files)
        X, y, testset = pdf_builder(files_test, filepath_to_dfs, label, feature_set, me_set, True)
        predictions = model_(X)
        testset['year'] = np.array(testset['date']).astype('datetime64[Y]').astype(int) + 1970
        mc_labels = []
        for i in range(10):
            mc_labels.append(i)
        if label == "ret":
            loss = nn.MSELoss(reduction = 'mean')
            predictions = predictions.reshape(-1)
            test_loss = loss(predictions, y)
            predictions = predictions.detach().numpy()
            r2 = r2_score(y, predictions)
            testset['predictions'] = predictions
            testset['ret_group'] = testset.groupby('date').predictions.transform(lambda x: pd.qcut(x.rank(method='first'), q = 10, labels = mc_labels)).astype(int)
            testset.to_csv(path_to_year + "/predictions.csv", index = False)

            testset['sq_error'] = (testset['predictions'] - testset['ret']) ** 2
            
            year_means = testset.groupby('year').ret.mean()
            
            
            testset = testset.join(year_means, on = 'year', rsuffix = '_mean')
            
            testset['ret_yvar'] = (testset['ret'] - 0) ** 2

            testset['sq_error'] = (testset['predictions'] - testset['ret']) ** 2
            
            r_2_years = pd.DataFrame(1 - (testset.groupby('year').sq_error.sum() / testset.groupby('year').ret_yvar.sum()))
            r_2_years['baseline'] = 0.0
            
            r_2_years.to_csv(path_to_year + "/r2.csv")
            # print(df)
            
            testset = testset[testset['ret_group'].isin([0, 9])]
            
            
            short = testset[testset['ret_group'] == 0]
            long = testset[testset['ret_group'] == 9]
        
            
        if label == "ret_binary":
            loss = nn.BCELoss(reduction = 'mean')
            predictions = predictions.reshape(-1)
            test_loss = loss(predictions, y)
            predictions = predictions.detach().numpy()
            testset['predictions'] = predictions
            testset['ret_group'] = testset.groupby('date').predictions.transform(lambda x: pd.qcut(x.rank(method='first'), q = 10, labels = mc_labels)).astype(int)
            testset.to_csv(path_to_year + "/predictions.csv", index = False)

            
            testset['binary_pred'] = np.where(testset.predictions < 0.5 ,0, 1)
            testset['right_class'] = np.where(testset.ret_binary == testset.binary_pred,1, 0)
            
            acc_year = testset.groupby('year')[['right_class', 'ret_binary']].mean()
            
            testset = testset[testset['ret_group'].isin([0, 9])]
            acc_year_HL = testset.groupby('year').right_class.mean()
            
            short = testset[testset['ret_group'] == 0]
            acc_year_L = short.groupby('year').right_class.mean()
            long = testset[testset['ret_group'] == 9]
            acc_year_H = long.groupby('year').right_class.mean()
            
            acc_year.to_csv(path_to_year + "/acc_year.csv")
            acc_year_HL.to_csv(path_to_year + "/precision_year_hl.csv")
            acc_year_L.to_csv(path_to_year + "/precision_year_l.csv")
            acc_year_H.to_csv(path_to_year + "/precision_year_h.csv")
            
            print(acc_year_HL)
            
        if label == "ret_multiclass":
            loss = nn.CrossEntropyLoss(reduction = 'mean')
            y = y.type(torch.LongTensor)
            test_loss = loss(predictions, y)
            
            testset_tmp = testset.copy()
            testset_tmp[mc_labels] = predictions
            
            testset_tmp.to_csv(path_to_year + "/predictions.csv", index = False)
            
            
            predictions = torch.max(predictions, dim=-1, keepdim=True)[1].flatten()
            predictions = predictions.detach().numpy()
            
            
            
            
            testset['predictions'] = predictions
            testset['ret_group'] = predictions
            # testset.to_csv(path_to_year + "/predictions.csv", index = False)

            testset['right_class'] = np.where(testset.ret_multiclass == testset.predictions,1, 0)
            acc_year = testset.groupby('year').right_class.mean()
            
            # print(df)
            
            testset['ret_group'] = testset.groupby('date').predictions.transform(lambda x: pd.qcut(x.rank(method='first'), q = 10, labels = mc_labels)).astype(int)
            
            
            
            testset = testset[testset['ret_group'].isin([0, 9])]
            acc_year_HL = testset.groupby('year').right_class.mean()
            
            short = testset[testset['ret_group'] == 0]
            acc_year_L = short.groupby('year').right_class.mean()
            
            
            long = testset[testset['ret_group'] == 9]
            acc_year_H = long.groupby('year').right_class.mean()
            
            acc_year.to_csv(path_to_year + "/acc_year.csv")
            acc_year_HL.to_csv(path_to_year + "/precision_year_hl.csv")
            acc_year_L.to_csv(path_to_year + "/precision_year_l.csv")
            acc_year_H.to_csv(path_to_year + "/precision_h.csv")
        

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
        port_year = portfolio[portfolio['year'] == start_test_year]
        
        portfolio.to_csv(path_to_year + "/portfolio.csv", index = False)
        port_year.to_csv(path_to_year + "/port_year.csv", index = False)
        config.to_csv(path_to_year + "/config.csv", index = False)
    configs = configs.drop(columns=['optimizer'])
    configs.to_csv("D:/Koen/Msc Data Science & Society/Thesis/" + ap + "_configs.csv")
    del configs