from enum import Enum,unique
from torch.utils.data import DataLoader
import torch
from tqdm.notebook import tqdm

@unique
class TrainState(Enum):
    TRAIN = 0
    TEST = 2

class Trainer():
    def __init__(self,model,device,lr = 0.0006,momentum=0.9,total_epoch = 10, weight_decay = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.current_epoch = 0
        self.total_epoch = total_epoch
        self.wight_decay = weight_decay
        self.models = {"hw2Model":model}
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.models["hw2Model"].parameters(), lr=lr,momentum=momentum)
        self.device = device
        self.models["hw2Model"].to(self.device)
        self.criterion.to(self.device)
        print(f'Optimizer: {self.optimizer}')
        print(f'Device: {self.optimizer}')
        self.dataloader_train = None
        self.dataloader_test = None

    def set_dataloader(self,train,test):
        self.dataloader_train = train
        self.dataloader_test = test

    def accuracy(self,source,target):
        source = source.max(1)[1].long().cpu()
        target = target.cpu()
        correct = (source == target).sum().item()
        return correct / float(source.shape[0])

    def fit(self):
        for _ in range(self.current_epoch,self.total_epoch):
            loss, acc = self.run_epoch(self.dataloader_train,TrainState.TRAIN)
            self.current_epoch += 1
        with torch.no_grad():
            test_loss,test_acc = self.run_epoch(self.dataloader_test,TrainState.TEST)
        print(f'test_loss={test_loss:.5f}, test_acc={test_acc * 100:.2f}%')

    def run_epoch(self, dataloader: DataLoader, train_state:TrainState):
        if train_state == TrainState.TRAIN:
            self.models["hw2Model"].train()
            desc_message = f'Epoch {self.current_epoch:02}'
        else:
            self.models["hw2Model"].eval()
            desc_message = f'Test'

        epoch_loss = 0
        epoch_acc = 0
        pbar = tqdm(dataloader, desc=desc_message)
        num_data = 0
        for input,label in pbar:
            input = input.to(self.device)
            label = label.to(self.device)

            prediction = self.models["hw2Model"](input)
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






