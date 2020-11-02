import os
import numpy as np
from tqdm import tqdm
import librosa
from musicnn.extractor import extractor

class Preprocessing():
    def __init__(self,file_list_path):
        self.file_list_path = file_list_path
        self.sampling_rate = 16000
        self.hop_length = 512
        self.number_fft = 1024
        self.number_mel = 96
        self.genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae']
        self.genres_dict = {genre_name: label_num for label_num, genre_name in enumerate(self.genres)}
        self.train_path_list = self.load_data_path(os.path.join(self.file_list_path,'split/train.txt'))
        self.test_path_list = self.load_data_path(os.path.join(self.file_list_path,'split/test.txt'))

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
                _, _, embeds = extractor(f'{self.file_list_path}/wav/{path_in}', model='MTT_musicnn', extract_features=True)
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
    
    def get_train_path_list(self,musiccnn=False):
        path_out_list = []
        feature_name = '/spec/' if musiccnn == False else '/embed/'
        for path_in in tqdm(self.train_path_list):
            path_out = self.file_list_path + feature_name + path_in.replace('.wav','.npy')
            path_out_list.append(path_out)
        return path_out_list

    def get_test_path_list(self,musiccnn=False):
        path_out_list = []
        feature_name = '/spec/' if musiccnn == False else '/embed/'
        for path_in in tqdm(self.test_path_list):
            path_out = self.file_list_path + feature_name + path_in.replace('.wav','.npy')
            path_out_list.append(path_out)
        return path_out_list