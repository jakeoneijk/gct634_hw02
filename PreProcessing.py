import os
import numpy as np
from tqdm import tqdm
import random
import librosa
import math
import pickle
from musicnn.extractor import extractor

class Preprocessing():
    def __init__(self,file_list_path,valid_ration= 0.1):
        self.file_list_path = file_list_path
        self.sampling_rate = 16000
        self.hop_length = 512
        self.number_fft = 1024
        self.number_mel = 96
        self.genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae']
        self.genres_dict = {genre_name: label_num for label_num, genre_name in enumerate(self.genres)}
        self.train_path_list = self.load_data_path(os.path.join(self.file_list_path,'split/train.txt'))
        self.test_path_list = self.load_data_path(os.path.join(self.file_list_path,'split/test.txt'))
        self.valid_ratio = 0.1
        self.chunk_sec = 5

    def load_data_path(self,path):
        with open(path) as f:
            file_paths = [line.rstrip('\n') for line in f]
        return file_paths
    
    def preprocess(self,musiccnn=False):
        debug_message = "preprocess" if musiccnn == False else "musiccnn preprocess"
        print(f'{debug_message} start')
        feature_name = '/spec/' if musiccnn == False else '/embed/'
        #Make dir for mel spec
        for genre in self.genres:
            os.makedirs(self.file_list_path + feature_name+genre, exist_ok=True)
        path_out_list = []
        for path_in in tqdm(self.train_path_list + self.test_path_list):
            path_out = self.file_list_path + feature_name + path_in.replace('.wav','.npy')
            path_out_list.append(path_out)

            if os.path.isfile(path_out):
                print(f'{path_out} already exists')
                continue
            if musiccnn:
                _, _, embeds = extractor(f'{self.file_list_path}/wav/{path_in}', model='MSD_musicnn', extract_features=True)
                embed = embeds['max_pool'].mean(axis=0)
                np.save(path_out, embed)
            else:
                signal, _ = librosa.load(f'{self.file_list_path}/wav/{path_in}',sr=self.sampling_rate) 
                melspec = librosa.feature.melspectrogram(signal,sr=self.sampling_rate,n_fft=self.number_fft,
                                                    hop_length=self.hop_length,n_mels=self.number_mel)
                melspec = librosa.power_to_db(melspec)
                melspec = melspec.astype('float32')
                np.save(path_out, melspec)
        print(f'{debug_message} finish')
        return path_out_list
    
    def get_path_list(self,musiccnn=False):
        train_path_out_list = []
        valid_path_out_list = []
        test_path_out_list = []
        number_of_train_valid = len(self.train_path_list)
        train_valid_indicies = list(range(number_of_train_valid))
        random.shuffle(train_valid_indicies)
        train_idc = train_valid_indicies[math.floor(number_of_train_valid*self.valid_ratio):]
        
        feature_name = '/spec/' if musiccnn == False else '/embed/'
        for path_in in tqdm(self.test_path_list):
            path_out = self.file_list_path + feature_name + path_in.replace('.wav','.npy')
            test_path_out_list.append(path_out)

        for i,path_in in enumerate(tqdm(self.train_path_list)):
            path_out = self.file_list_path + feature_name + path_in.replace('.wav','.npy')
            if i in train_idc:
                train_path_out_list.append(path_out)
            else:
                valid_path_out_list.append(path_out)
        return train_path_out_list,valid_path_out_list,test_path_out_list
    
    def preprocess_chunk(self,musiccnn=False):
        debug_message = "preprocess chunk" if musiccnn == False else "musiccnn preprocess chunk"
        print(f'{debug_message} start')
        feature_name = f'/spec_chunk{self.chunk_sec}' if musiccnn == False else f'/embed_chunk{self.chunk_sec}'
        train_dir = self.file_list_path + feature_name +"/train"
        test_dir = self.file_list_path + feature_name +"/test"
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir,exist_ok=True)
        os.makedirs(self.file_list_path + feature_name+"/augmentaion", exist_ok=True)
        self.preprocess_chunk_for_given_path(train=True,save_path=train_dir,musiccnn=musiccnn)
        self.preprocess_chunk_for_given_path(train=False,save_path=test_dir,musiccnn=musiccnn)
    
    def preprocess_chunk_for_given_path(self,train:bool,save_path:str,musiccnn=False):
        path_list = self.train_path_list if train else self.test_path_list
        for path_in in tqdm(path_list):
            genre = path_in.split('/')[0]
            file_name = path_in.split('/')[1].replace(".wav","")
            if os.path.isfile(os.path.join(save_path,f'{file_name}_0.pkl')):
                print(f'pickle data of {file_name} already exists')
                continue
            print("debug")
            signal, _ = librosa.load(f'{self.file_list_path}/wav/{path_in}',sr=self.sampling_rate)
            self.preprocess_chunk_save(train,signal,genre=genre,
                                                    save_file_path=os.path.join(save_path,f'{file_name}'),musiccnn=musiccnn)
            if train == True:
                noise_signal = self.data_augmentation_noise(signal)
                shift_signal = self.data_augmentation_shift(signal)
                stretch_signal = self.data_augmentation_stretch(signal)
            
                self.preprocess_chunk_save(train,noise_signal,genre=genre,
                                                    save_file_path=os.path.join(save_path,f'{file_name}'),musiccnn=musiccnn,
                                                    data_augmentation=True,data_augmentaion_type="noise")
                self.preprocess_chunk_save(train,shift_signal,genre=genre,
                                                    save_file_path=os.path.join(save_path,f'{file_name}'),musiccnn=musiccnn,
                                                    data_augmentation=True,data_augmentaion_type="shift")
                self.preprocess_chunk_save(train,stretch_signal,genre=genre,
                                                    save_file_path=os.path.join(save_path,f'{file_name}'),musiccnn=musiccnn,
                                                    data_augmentation=True,data_augmentaion_type="stretch")
                
    def preprocess_chunk_save(self,train,signal,genre:str,save_file_path,musiccnn=False,data_augmentation=False,data_augmentaion_type=""):
        pickle_idx = 0
        test_feature_array = []
        for start_idx in range(0,len(signal),self.sampling_rate*self.chunk_sec):
            end_idx = start_idx + self.sampling_rate*self.chunk_sec
            music_chunk = signal[start_idx:end_idx]
            if music_chunk.size != (self.sampling_rate*self.chunk_sec):
                continue

            feature = self.feature_extraction(music_chunk,musiccnn)
            if train == True:
                save_file_name = save_file_path+f'_{pickle_idx}.pkl'
                if data_augmentation == True:
                    dir_array = save_file_path.split("/")
                    save_file_name = dir_array[0] +"/"+ dir_array[1] +"/"+ dir_array[2]+"/augmentaion/"+ dir_array[4]+"("+data_augmentaion_type+")"+f'_{pickle_idx}.pkl'
                print(f'Saving: {save_file_name}')
                with open (save_file_name,'wb') as writing_file:
                    pickle.dump({"feature":feature,"genre":genre},writing_file)
            else:
                test_feature_array.append({"feature":feature,"genre":genre})
            pickle_idx += 1
             
        if train==False:
            save_file_name = save_file_path+'_0.pkl'
            print(f'Saving: {save_file_name}')
            with open (save_file_name,'wb') as writing_file:
                pickle.dump(test_feature_array,writing_file)

    def feature_extraction(self,signal,musiccnn=False):
        melspec = librosa.feature.melspectrogram(signal,sr=self.sampling_rate,n_fft=self.number_fft,
                                                    hop_length=self.hop_length,n_mels=self.number_mel)
        melspec = librosa.power_to_db(melspec)
        melspec = melspec.astype('float32')
        return melspec
        
    def data_augmentation_noise(self,signal):
        noise = np.random.randn(len(signal))
        signal_noise = signal + 0.005 * noise
        return signal_noise
    
    def data_augmentation_shift(self,signal):
        return np.roll(signal, 1600)
    
    def data_augmentation_stretch(self,signal, rate=1):
        return librosa.effects.time_stretch(signal, rate)


        