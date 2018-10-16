from csv import reader
from CNNmodel import Net
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
import sys
import SavLoad as sl
import os
import gc
#read the data
def readdata(filename):
  dataset=list()
  with open(filename,'r') as file:
    csv_reader=reader(file)
    for row in csv_reader:
      if not row:
        continue
      dataset.append(row)
  return dataset

def str_column_to_float(dataset,column):
  for row in dataset:
    row[column]=float(row[column])

def str_column_to_int(dataset,column):
  for row in dataset:
    row[column]=int(row[column])
def validate(model,eval_data,criterion,eval_labels):
  running_loss=0.0
  correct=0
  for i in range(len(eval_data)):
    #get the inputs
    inputs, labels = eval_data[i],eval_labels[i]
    #Converting To Tensors
    inputs=torch.from_numpy(inputs).float()
    #labels=torch.from_numpy(labels)
    # wrap them in Variable
    inputs = Variable(inputs)

    # forward 
    outputs = model(inputs)
    #Predicted class
    _, predicted = torch.max(outputs.data, 1)
    correct += (predicted == labels).sum()
    #loss = criterion(outputs, labels)
    #running_loss=running_loss+loss
  return correct/i


def main():
  #Loading Data
  file="/DeepLearning/Code/char_test_26_all_new_32*32_rotate.csv"
  dataset=readdata(file)
  for i in range(len(dataset[0])-1):
    str_column_to_float(dataset,i)
  str_column_to_int(dataset,len(dataset[0])-1)
  train_data=[]
  train_labels=[]
  eval_data=[]
  eval_labels=[]
  for i in range(len(dataset)):
    if (i%20)<12:
      train_data.append(np.array(dataset[i][:-1]))
      train_labels.append(dataset[i][-1])
    else:
      eval_data.append(np.array(dataset[i][:-1]))
      eval_labels.append(dataset[i][-1])
  #Training Data For Pytorch 
  #print(len(train_data))
  train_data=np.reshape(train_data,(-1,1,32,32))
  eval_data=np.reshape(eval_data,(-1,1,32,32))
  eval_data=np.array_split(eval_data,len(eval_data))
  train_data=np.array_split(train_data,150)
  train_labels=np.array_split(train_labels,150)

  #Creating a new Model
  model=Net()
  
  #Loss Function
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.3, momentum=0.9)
  start_epoch=0
  max_epoch=3000
  best_prec=0
  Pathname='model_best.pth.tar'

  #Loading the Checkpoint
  if os.path.isfile(Pathname):
    print("=> loading checkpoint '{}'".format(Pathname))
    start_epoch,best_prec,model,optimizer=sl.LoadModel(Pathname,model,optimizer,start_epoch,best_prec)
  else:
      print("=> no checkpoint found at '{}'".format(Pathname))
  #Train
  for epoch in range(start_epoch,max_epoch):  # loop over the dataset multiple times
    print(epoch)
    running_loss = 0.0
    for i in range(len(train_data)):
        # get the inputs
        print(i)
        inputs, labels = train_data[i],train_labels[i]
        #Converting To Tensors
        inputs=torch.from_numpy(inputs).float()
        labels=torch.from_numpy(labels)
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        is_best=2
        running_loss=running_loss+loss
        gc.collect()

    print("Epoch: ",epoch,"Loss:",running_loss)

    if(epoch%5==0):
      best_prec1=validate(model,eval_data,criterion,eval_labels)
      sl.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer' : optimizer.state_dict(),
        }, is_best)
      print("Accuracy",best_prec1,"Loss",running_loss)
  print('Finished Training')




main()
  
