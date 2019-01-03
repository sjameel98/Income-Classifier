import argparse
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import MultiLayerPerceptron
from dataset import AdultDataset
from util import *

import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal
np.random.seed(0)
torch.manual_seed(0)


""" Adult income classification

In this lab we will build our own neural network pipeline to do classification on the adult income dataset. More
information on the dataset can be found here: http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html

"""
seed = 0

# =================================== LOAD DATASET =========================================== #

######

# 2.1 YOUR CODE HERE
data = pd.read_csv('data/adult.csv')
######

# =================================== DATA VISUALIZATION =========================================== #

# the dataset is imported as a DataFrame object, for more information refer to
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
# we can check the number of rows and columns in the dataset using the .shape field
# to get a taste of what our datset looks like, let's visualize the first 5 rows of the dataset using the .head() method
# the task will be to predict the "income" field (>50k or <50k) based on the other fields in the dataset
# check how balanced our dataset is using the .value_counts() method.

######

# 2.2 YOUR CODE HERE
print(data.shape)
print(data.columns)
verbose_print(data.head())
print(data["income"].value_counts())
######


# =================================== DATA CLEANING =========================================== #

# datasets often come with missing or null values, this is an inherent limit of the data collecting process
# before we run any algorithm, we should clean the data of any missing values or unwanted outliers which could be
# detrimental to the performance or training of the algorithm. In this case, we are told that missing values are
# indicated with the symbol "?" in the dataset

# let's first count how many missing entries there are for each feature
col_names = data.columns
num_rows = data.shape[0]
for feature in col_names:
    pass
    ######

    # 2.3 YOUR CODE HERE
    a = data[feature].isin(["?"]).sum()
    print(a, 'missing values in ', feature)
    ######

# next let's throw out all rows (samples) with 1 or more "?"
# Hint: take a look at what data[data["income"] != ">50K"] returns
# Hint: if data[field] do not contain strings then data["income"] != ">50K" will return an error

    ######

    # 2.3 YOUR CODE HERE
for feature in col_names:
    a = data[feature].isin(["?"]).sum()
    if a > 0:
        data = data[data[feature] != "?"]
print('Size after removing missing values = ',data.shape)
    ######

# =================================== BALANCE DATASET =========================================== #

    ######

    # 2.4 YOUR CODE HERE
salarycounts = data["income"].value_counts()
lowestcount = min(salarycounts[0],salarycounts[1])
df1 = data[data['income'] == '<=50K'].sample(n = lowestcount, random_state = 0)
df2 = data[data['income'] == '>50K'].sample(n = lowestcount, random_state = 0)
frames = [df1, df2]
data = pd.concat(frames)
print('Size after removing random samples = ', data.shape)

    ######

# =================================== DATA STATISTICS =========================================== #

# our dataset contains both continuous and categorical features. In order to understand our continuous features better,
# we can compute the distribution statistics (e.g. mean, variance) of the features using the .describe() method

######

# 2.5 YOUR CODE HERE
verbose_print(data.describe())
######

# likewise, let's try to understand the distribution of values for discrete features. More specifically, we can check
# each possible value of a categorical feature and how often it occurs
categorical_feats = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']

for feature in categorical_feats:
    pass
    ######

    # 2.5 YOUR CODE HERE
    print(data[feature].value_counts())
    ######

# visualize the first 3 features using pie and bar graphs

######

# 2.5 YOUR CODE HERE
for i in range(3):
    pass
    #pie_chart(data,categorical_feats[i])
    #binary_bar_chart(data, categorical_feats[i])
######

# =================================== DATA PREPROCESSING =========================================== #

# we need to represent our categorical features as 1-hot encodings
# we begin by converting the string values into integers using the LabelEncoder class
# next we convert the integer representations into 1-hot encodings using the OneHotEncoder class
# we don't want to convert 'income' into 1-hot so let's extract this field first
# we also need to preprocess the continuous features by normalizing against the feature mean and standard deviation
# don't forget to stitch continuous and cat features together

# NORMALIZE CONTINUOUS FEATURES
######

# 2.6 YOUR CODE
continuous_feats = []
for feature in col_names:
    if feature not in categorical_feats:
        continuous_feats.append(feature)
continuous_df = (data[continuous_feats]-data[continuous_feats].mean())/(data[continuous_feats].std())
cont_array = continuous_df.values
######

# ENCODE CATEGORICAL FEATURES
######

# 2.6 YOUR CODE HERE
dflabel = data[categorical_feats]
label_encoder = LabelEncoder()
for feature in categorical_feats:
    dflabel[feature] = label_encoder.fit_transform(dflabel[feature])
######

oneh_encoder = OneHotEncoder()
######

# 2.6 YOUR CODE HERE

income_values = dflabel['income'].values
dflabel.drop('income', axis = 1, inplace=True)
categorical_feats.remove('income')
cat_data = oneh_encoder.fit_transform(dflabel[categorical_feats].values).toarray()
data_array = np.concatenate((cat_data, cont_array), axis = 1)
print(data_array.shape)
######
# Hint: .toarray() converts the DataFrame to a numpy array


# =================================== MAKE THE TRAIN AND VAL SPLIT =========================================== #
# we'll make use of the train_test_split method to randomly divide our dataset into two portions
# control the relative sizes of the two splits using the test_size parameter

######

# 2.7 YOUR CODE HERE
feat_train, feat_valid, label_train, label_valid = train_test_split(data_array, income_values.reshape(income_values.shape[0],1), test_size=0.2,random_state = 0)
print(feat_train.shape, feat_valid.shape, label_train.shape, label_valid.shape)
######

# =================================== LOAD DATA AND MODEL =========================================== #

def load_data(batch_size):
    ######

    # 3.2 YOUR CODE HERE
    train_dataset = AdultDataset(feat_train,label_train)
    val_dataset = AdultDataset(feat_valid,label_valid)
    ######
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle= True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, val_loader


def load_model(lr,hidden_layer,activation):

    ######

    # 3.4 YOUR CODE HERE
    model = MultiLayerPerceptron(feat_train.shape[1],hidden_layer,activation)
    loss_fnc = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    ######

    return model, loss_fnc, optimizer


def evaluate(model, val_loader):
    total_corr = 0

    ######

    # 3.6 YOUR CODE HERE
    for i, vbatch in enumerate(val_loader):
        feats, label = vbatch
        prediction = model(feats.float())
        corr = ((prediction>0.5) == (label>0.5)).squeeze()
        total_corr += int(corr.sum())
    ######

    return float(total_corr)/len(val_loader.dataset)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=10)
    #parser.add_argument('--hidden_layer', type=int, default=64)
    parser.add_argument('-l', '--hlsandsize', type=int, nargs='+', default = None)
    parser.add_argument('--activation', type=str, default = 'Tanh')

    args = parser.parse_args()

    ######

    # 3.5 YOUR CODE HERE
    bs = args.batch_size
    lrate = args.lr
    eps = args.epochs
    hlsize = args.hlsandsize
    activation = args.activation
    train_loader, val_loader = load_data(bs)
    model, loss_fnc, optimizer = load_model(lrate, hlsize,activation)

    t = 0
    start_time = time()
    '''train_plot = np.zeros((args.eval_every,))
    val_plot = np.zeros((args.eval_every,))
    step_plot = np.zeros((args.eval_every,))'''
    rem = int((17932-17932%bs)/bs) + int(17932%bs>0)
    max_val = 0
    max_step = 0
    train_list = []
    val_list = []
    steptrain_list = []
    stepval_list = []
    start_time = time()
    for epoch in range(eps):
        accum_loss = 0
        tot_corr = 0
        for i, batch in enumerate(train_loader):
            feats, label = batch
            optimizer.zero_grad()

            predictions = model(feats.float())
            batch_loss = loss_fnc(input=predictions.squeeze(), target = label.squeeze().float())
            accum_loss += batch_loss
            batch_loss.backward()
            optimizer.step()
            corr = ((predictions > 0.5) == (label > 0.5)).squeeze()
            tot_corr += int(corr.sum())

            if (t >= (eps*rem)-args.eval_every):
                train_list.append(float(tot_corr) / len(train_loader.dataset))
                #steptrain_list.append(t)
                steptrain_list.append(t+1)
            if (t+1)% args.eval_every == 0:
                valid_acc = evaluate(model,val_loader)
                val_list.append(valid_acc)
                #stepval_list.append(t)
                ourtime = time() -start_time
                stepval_list.append(t+1)
                print("Epoch: {}, Step {} | Loss: {}| Test acc:{}".format(epoch+1,t+1,accum_loss/100,valid_acc))
                if (valid_acc > max_val):
                    max_val = valid_acc
                    max_step = t+1
                accum_loss = 0
                '''if t >= (eps*rem)-args.eval_every:
                train_plot[t-((eps*rem)-args.eval_every)] = float(tot_corr)/len(train_loader.dataset)
                val_plot[t-((eps*rem)-args.eval_every)] = valid_acc
                step_plot[t-((eps*rem)-args.eval_every)] = t'''
            t = t+1
    val_plot = np.array(val_list)
    steptrain_plot = np.array(steptrain_list)
    stepval_plot = np.array(stepval_list)
    train_plot = np.array(train_list)
    print("Train acc: {}".format(float(tot_corr)/len(train_loader.dataset)))
    end_time = time()
    elapsed_time = end_time - start_time
    print('Elapsed Time = ', elapsed_time)
    print('Max Validation Accuracy = ', max_val, ' at Step ', max_step)
    plt.subplot(2,1,1)
    plt.plot(stepval_plot, sp.signal.savgol_filter(val_plot, 2*bs+1,5))
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title("Validation Accuracy for Learning Rate = {} | Batch Size = {} | Hidden Layer Size = {}| Epochs = {}| Activation = {}".format(lrate, bs, hlsize, eps,activation))
    plt.legend(['Validation'])

    plt.subplot(2,1,2)
    plt.plot(steptrain_plot, sp.signal.savgol_filter(train_plot,9,5))
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title("Training Accuracy for Learning Rate = {} | Batch Size = {} | Hidden Layer Size = {}| Epochs = {}".format(lrate, bs, hlsize, eps))
    plt.legend(['Train'])
    plt.show()
    ######


if __name__ == "__main__":
    main()