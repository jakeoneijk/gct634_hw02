import random
import math
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

class DataloaderBuilderChunk():
    def __init__(self,train_dset:Dataset,test_dset:Dataset,batch_size,num_workers=1,valid_ration=0.1):
        self.train_dataset = train_dset
        self.test_dataset = test_dset
        self.batch_size = batch_size
        self.num_workers = num_workers

        total_train_dset_num = len(train_dset)
        total_train_indicies = list(range(total_train_dset_num))
        random.shuffle(total_train_indicies)
        self.valid_indicies = total_train_indicies[:math.floor(total_train_dset_num*valid_ration)]
        self.train_indicies = total_train_indicies[math.floor(total_train_dset_num*valid_ration):]
        print("debug")
    
    def get_data_loader(self):
        train_dataloader = DataLoader(self.train_dataset,sampler=SubsetRandomSampler(self.train_indicies)
                                        ,pin_memory=True,drop_last=True,num_workers=self.num_workers,batch_size=self.batch_size)
        valid_dataloader = DataLoader(self.train_dataset,sampler=SubsetRandomSampler(self.valid_indicies)
                                        ,pin_memory=True,drop_last=True,num_workers=self.num_workers,batch_size=self.batch_size)
        test_dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, drop_last=False)
        return train_dataloader,valid_dataloader,test_dataloader
        
