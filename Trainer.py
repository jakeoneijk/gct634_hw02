from enum import Enum,unique
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
import numpy as np
import sklearn

@unique
class TrainState(Enum):
    TRAIN = 0
    VALID = 1
    TEST = 2

class Trainer():
    def __init__(self,model,device,num_input = 1,lr = 0.0001,momentum=0.9,total_epoch = 10, weight_decay = 0.0):
        self.num_input = num_input
        self.lr = lr
        self.momentum = momentum
        self.current_epoch = 0
        self.total_epoch = total_epoch
        self.wight_decay = weight_decay
        self.models = {"hw2Model":model}
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.models["hw2Model"].parameters(), lr=lr) #torch.optim.SGD(self.models["hw2Model"].parameters(), lr=lr,momentum=momentum)
        self.device = device
        self.models["hw2Model"].to(self.device)
        self.criterion.to(self.device)
        print(f'Optimizer: {self.optimizer}')
        print(f'Device: {self.optimizer}')
        self.dataloader_train = None
        self.dataloader_valid = None
        self.dataloader_test = None
        self.model_save_path = f'./ModelSave/best_{self.models["hw2Model"].__class__.__name__}.pth'
        self.best_model_epoch = 0
        self.best_model_loss = 5000
        self.best_model_accu = 0
        self.y_array = []
        self.y_pred_array = []

    def set_dataloader(self,train,valid,test):
        self.dataloader_train = train
        self.dataloader_valid = valid
        self.dataloader_test = test

    def accuracy(self,source,target):
        source = source.max(1)[1].long().cpu()
        target = target.cpu()
        correct = (source == target).sum().item()
        return correct / float(source.shape[0])

    def fit(self,chunk=False):
        self.y_array = []
        self.y_pred_array = []
        print("\n===================train start===================")
        print(f'==================={self.models["hw2Model"].__class__.__name__}===================')
        for _ in range(self.current_epoch,self.total_epoch):
            loss, acc = self.run_epoch(self.dataloader_train,TrainState.TRAIN)
            with torch.no_grad():
                valid_loss, valid_acc = self.run_epoch(self.dataloader_valid,TrainState.VALID)
                self.save_best_model(current_loss=valid_loss,current_accu=valid_acc)
            self.current_epoch += 1

        best_model_load = torch.load(self.model_save_path)
        self.models["hw2Model"].load_state_dict(best_model_load)

        if chunk == True:
            with torch.no_grad():
                test_acc = self.test_by_chunk(self.dataloader_test,TrainState.TEST)
        else:
            with torch.no_grad():
                _,test_acc = self.run_epoch(self.dataloader_test,TrainState.TEST)
        print(f'{self.models["hw2Model"].__class__.__name__}: test_acc={test_acc * 100:.2f}%')
        return test_acc, sklearn.metrics.confusion_matrix(self.y_array,self.y_pred_array)

    def run_epoch(self, dataloader: DataLoader, train_state:TrainState):
        if train_state == TrainState.TRAIN:
            self.models["hw2Model"].train()
            desc_message = f'Epoch {self.current_epoch:02}'
        else:
            self.models["hw2Model"].eval()
            desc_message = f'Test {self.best_model_epoch} epoch model' if train_state != TrainState.VALID else f'Valid'

        epoch_loss = 0
        epoch_acc = 0
        pbar = tqdm(dataloader, desc=desc_message)
        num_data = 0
        for input,label in pbar:
            label = label.to(self.device)
            if self.num_input == 1:
                input = input.to(self.device)
                prediction = self.models["hw2Model"](input)
            else:
                input_1 = input["spec"].to(self.device)
                input_2 = input["embed"].to(self.device)
                prediction = self.models["hw2Model"](input_1,input_2)
            loss = self.criterion(prediction,label)
            acc = self.accuracy(prediction,label)

            if train_state == TrainState.TRAIN:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            batch_size = len(input)
            num_data += batch_size
            epoch_loss += batch_size * loss.item()
            epoch_acc += batch_size * acc
            pbar.set_postfix({'loss': epoch_loss / num_data,
                              'acc': epoch_acc / num_data})
        return (epoch_loss / num_data),(epoch_acc / num_data)

    def save_best_model(self, current_loss, current_accu):
        #if current_loss < self.best_model_loss and current_accu > self.best_model_accu:
        if current_accu > self.best_model_accu:
            print("save current best model")
            self.best_model_epoch = self.current_epoch
            self.best_model_loss = current_loss
            self.best_model_accu = current_accu
            torch.save(self.models["hw2Model"].state_dict(),self.model_save_path)

    def test_by_chunk(self, dataloader: DataLoader, train_state:TrainState):
        print(f'Test {self.best_model_epoch} epoch model')
        self.models["hw2Model"].eval()
        total_num_data = len(dataloader)
        correct_num = 0
        for input,label in dataloader:
            if self.num_input == 1:
                input = input.view(input.shape[1],input.shape[2],input.shape[3])
                input = input.to(self.device)
                prediction = self.models["hw2Model"](input)
            else:
                input_1 = input["spec"].view(input["spec"].shape[1],input["spec"].shape[2],input["spec"].shape[3])
                input_2 = input["embed"].view(input["embed"].shape[1],input["embed"].shape[2])
                input_1 = input_1.to(self.device)
                input_2 = input_2.to(self.device)
                prediction = self.models["hw2Model"](input_1,input_2)
                
            label = label.to(self.device)
            prediction = prediction.max(1)[1].long().cpu()
            prediction = prediction.numpy()
            count = np.bincount(prediction)
            prediction_label = count.argmax()

            self.y_array.append(label[0])
            self.y_pred_array.append(prediction_label)

            if label[0] == prediction_label:
                correct_num += 1
        return correct_num/total_num_data

    def only_test(self,model_path,dataloader,chunk,message=""):
        print("\n=======================================\n")
        print(message)
        best_model_load = torch.load(model_path)
        self.models["hw2Model"].load_state_dict(best_model_load)

        if chunk == True:
            with torch.no_grad():
                test_acc = self.test_by_chunk(dataloader,TrainState.TEST)
        else:
            with torch.no_grad():
                _,test_acc = self.run_epoch(dataloader,TrainState.TEST)
        print(f'{self.models["hw2Model"].__class__.__name__}: test_acc={test_acc * 100:.2f}%')






