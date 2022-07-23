import torch
import random
import librosa
import numpy as np
from glob import glob
from librosa.feature import delta
from torch.utils.data import Dataset
from librosa.feature.spectral import mfcc
from sklearn.preprocessing import LabelBinarizer

# Available phoneme classes

phoneme_set = ['aa', 'ae', 'ah', 'ay', 'aw', 'eh', 'er', 'ey', 'ih', 'iy', 'l',
     'ow', 'oy', 'r', 'uh', 'uw', 'w', 'y', 'p', 't', 'k', 'b', 'd', 'g', 'jh', 
     'ch', 's', 'sh', 'z', 'f', 'th', 'v', 'dh', 'hh', 'm', 'n', 'ng', 'sil']

# create clusters
phoneme_set_cluster = ['Cluster2', 'Cluster3','Cluster4', 'Cluster5', 'Cluster6',
     'Cluster7', 'ClusterS']

phone_cluster_trans = {
    'aa':'Cluster6', 'ae':'Cluster6', 'ah':'Cluster6','ay':'Cluster6','aw':'Cluster6','eh':'Cluster6', 'er':'Cluster6', 
    'ey':'Cluster6', 'ih':'Cluster6', 'iy':'Cluster6', 'l':'Cluster6',
     'ow':'Cluster6', 'oy':'Cluster6', 'r':'Cluster6', 'uh':'Cluster6', 'uw':'Cluster6', 'w':'Cluster6', 'y':'Cluster3', 
     'p':'Cluster2', 't':'Cluster2', 'k':'Cluster2', 'b':'Cluster2', 
     'd':'Cluster2', 'g':'Cluster2', 'jh':'Cluster7', 'ch':'Cluster7', 's':'Cluster7', 'sh':'Cluster7', 'z':'Cluster7', 
     'f':'Cluster2', 'th':'Cluster2', 'v':'Cluster2', 'dh':'Cluster2', 'hh':'Cluster4', 
     'm':'Cluster5', 'n':'Cluster5', 'ng':'Cluster5', 'sil':'ClusterS'
}
phoneme_set_clusterASR = ['Vowels', 'Diphthongs', 'Semi-vowels', 'Stops', 'Fricatives', 'Nasals', 'Silence']

phone_clusterASR_trans = {
    'aa':'Vowels', 'ae':'Vowels', 'ah':'Vowels','ay':'Diphthongs','aw':'Diphthongs','eh':'Vowels', 'er':'Semi-vowels', 
    'ey':'Diphthongs', 'ih':'Vowels', 'iy':'Vowels', 'l':'Semi-vowels',
     'ow':'Diphthongs', 'oy':'Diphthongs', 'r':'Semi-vowels', 'uh':'Vowels', 'uw':'Vowels', 'w':'Semi-vowels', 'y':'Semi-vowels', 
     'p':'Stops', 't':'Stops', 'k':'Stops', 'b':'Stops', 
     'd':'Stops', 'g':'Stops', 'jh':'Stops', 'ch':'Stops', 's':'Fricatives', 'sh':'Fricatives', 'z':'Fricatives', 
     'f':'Fricatives', 'th':'Fricatives', 'v':'Fricatives', 'dh':'Fricatives', 'hh':'Fricatives', 
     'm':'Nasals', 'n':'Nasals', 'ng':'Nasals', 'sil':'Silence'
}

# our viseme sets
viseme_set = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'S'] # hh is related to the next letter and because it is lip rounding based vowels we take I. 

# transition from phoneme to viseme
phone_vis_trans = {
    'aa':'I', 'ae':'I', 'ah':'I','ay':'I','aw':'D','eh':'I', 'er':'B', 'ey':'I', 'ih':'I', 'iy':'I', 'l':'J',
     'ow':'B', 'oy':'G', 'r':'B', 'uh':'B', 'uw':'B', 'w':'B', 'y':'I', 'p':'C', 't':'J', 'k':'K', 'b':'C', 'd':'J', 'g':'K', 'jh':'F', 
     'ch':'F', 's':'H', 'sh':'F', 'z':'H', 'f':'A', 'th':'E', 'v':'A', 'dh':'E', 'hh':'I', 'm':'C', 'n':'J', 'ng':'K', 'sil':'S'
}

viseme_set_new = ['A', 'C', 'E', 'F', 'H', 'I', 'J', 'K', 'S'] # remove H for 8

phone_vis_trans_9 = {
    'aa':'I', 'ae':'I', 'ah':'I','ay':'I','aw':'I','eh':'I', 'er':'I', 'ey':'I', 'ih':'I', 'iy':'I', 'l':'J',
     'ow':'I', 'oy':'I', 'r':'J', 'uh':'I', 'uw':'I', 'w':'K', 'y':'I', 'p':'C', 't':'J', 'k':'K', 'b':'C', 'd':'J', 'g':'K', 'jh':'F', 
     'ch':'F', 's':'H', 'sh':'F', 'z':'H', 'f':'A', 'th':'E', 'v':'A', 'dh':'E', 'hh':'I', 'm':'C', 'n':'J', 'ng':'K', 'sil':'S'
}
phone_vis_trans_8 = {
    'aa':'I', 'ae':'I', 'ah':'I','ay':'I','aw':'I','eh':'I', 'er':'I', 'ey':'I', 'ih':'I', 'iy':'I', 'l':'J',
     'ow':'I', 'oy':'I', 'r':'J', 'uh':'I', 'uw':'I', 'w':'K', 'y':'I', 'p':'C', 't':'J', 'k':'K', 'b':'C', 'd':'J', 'g':'K', 'jh':'F', 
     'ch':'F', 's':'J', 'sh':'F', 'z':'J', 'f':'A', 'th':'E', 'v':'A', 'dh':'E', 'hh':'I', 'm':'C', 'n':'J', 'ng':'K', 'sil':'S'
}

# dataloader for training and validation
class NTCDTIMIT(Dataset):
    def __init__(self, input_audio_dir, dataset_type, noise_dir, val_dir, video_dir, labels_dir, val = False, audio = False, video = False, mfccs = False, noise_types = ['Babble', 'Cafe', 'Car', 'LR', 'Street', 'White'], snrs = ['-5', '0', '5', '10', '15', '20']):
        # Do not open hdf5 in __init__ if num_workers > 0
        self.input_audio_dir = input_audio_dir
        self.dataset_type = dataset_type
        self.noise_dir = noise_dir
        self.val_dir = val_dir
        self.video_dir = video_dir
        self.labels_dir = labels_dir
        self.noise_types = noise_types
        self.snrs = snrs
        self.audio = audio
        self.video = video
        self.mfcc = mfccs
        self.val = val

        # List of files and get short paths
        noisy_audio_paths = sorted(glob(self.input_audio_dir + self.dataset_type + '/**/*.wav', recursive=True))
        self.file_paths = [paths[len(self.input_audio_dir + self.dataset_type):] for paths in noisy_audio_paths]
        
        # use label binarizer for one hot encoding for phoneme labels
        self.lb = LabelBinarizer().fit(phoneme_set)
        self.lbViseme = LabelBinarizer().fit(viseme_set)
        
        self.dataset_len = len(self.file_paths)

    # z-normalization for audio data (subtract mean and devide by variance)
    def normalisation(self, i):
        i_std = np.std(i)
        if i_std == 0.:
            i_std = 1.
        return (i - np.mean(i))/i_std

    def __getitem__(self, i):
        
        if self.audio or self.mfcc:
            # choose random noise_type and SNR
            noise_type = self.noise_types[random.randint(0, len(self.noise_types)-1)]
            snr = self.snrs[random.randint(0, len(self.snrs)-1)]

            # load speaker/utterance with random noise_type and SNR 
            noisy_speech = librosa.load(self.noise_dir + self.dataset_type + '/' + noise_type + '/'+ snr + self.file_paths[i], sr=16000)[0]

            # load clean speach
            if self.val:
                val_speech = librosa.load(self.val_dir + self.file_paths[i], sr=16000)[0]
                audio = self.normalisation(val_speech)
            # z-normalize audio
            else: 
                audio = self.normalisation(noisy_speech)

        if self.mfcc: 
            # use librosa mfcc and delta calculation to obtain the mfccs and their derivatives
            mfccs = mfcc(audio, sr=16000, hop_length=256, n_mfcc=13, win_length=1024)
            mfccs_v = delta(mfccs, 5)
            mfccs_acc = delta(mfccs_v, 5)

            mfccs_complete = np.transpose(np.concatenate((mfccs, mfccs_v, mfccs_acc)))

        if self.video:
            video = np.load(self.video_dir + self.dataset_type +self.file_paths[i][:-4] + '.npy')
            video = np.moveaxis(video, -1, 0)
            

        # load labels with a frame rate of 30 frames/s
        labels = np.load(self.labels_dir + self.dataset_type +self.file_paths[i][:-4] + '.npy')
        labels = np.array([n.encode("ascii", "ignore") for n in labels])
        
        if self.video and not (self.audio or self.mfcc):
            labels = [phone_vis_trans[str(label)[2:-1]] for label in labels]
            labels = self.lbViseme.transform(labels)
        else:
            labels = self.lb.transform(labels)

        if self.audio:
            if(len(audio)/256 < len(labels)):
                labels=labels[:int(len(audio)/256)]
            elif(len(audio)/256 > len(labels)):
                audio=audio[:len(labels)*256]

        if self.mfcc:
            if mfccs_complete.shape[0] > labels.shape[0]:
                mfccs_complete = mfccs_complete[:labels.shape[0], :]
            elif mfccs_complete.shape[0] < labels.shape[0]:
                labels = labels[:mfccs_complete.shape[0], :]

        if self.video:
            if video.shape[0] > labels.shape[0]:
                video = video[:labels.shape[0], :, :]

            elif video.shape[0] < labels.shape[0]:
                labels = labels[:video.shape[0], :]
                if self.mfcc:
                    mfccs_complete = mfccs_complete[:video.shape[0], :]
                if self.audio:
                    audio=audio[:len(labels)*256]

        if self.video:
            if self.audio:
                if self.mfcc:
                    return torch.from_numpy(audio), torch.from_numpy(video), torch.from_numpy(mfccs_complete), torch.from_numpy(labels)
                else:
                    return torch.from_numpy(audio), torch.from_numpy(video), torch.from_numpy(labels)
            elif self.mfcc: 
                return torch.from_numpy(video), torch.from_numpy(mfccs_complete), torch.from_numpy(labels)    
            else:
                return torch.from_numpy(video), torch.from_numpy(labels)   
        elif self.audio:
            return torch.from_numpy(audio), torch.from_numpy(labels)       
        elif self.mfcc:
            return torch.from_numpy(mfccs_complete), torch.from_numpy(labels)

        
    def __len__(self):
        return self.dataset_len

# dataloader for testing
class NTCDTIMITTEST(Dataset):
    def __init__(self, input_audio_dir, audio_dir, video_dir, labels_dir, audio = False, video = False, mfccs = False, snrs = ['-5', '0', '5', '10', '15', '20']):
        # Do not open hdf5 in __init__ if num_workers > 0
        self.input_audio_dir = input_audio_dir
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.labels_dir = labels_dir
        self.snrs = snrs
        self.audio = audio
        self.video = video
        self.mfcc = mfccs

        # List of files and get short paths
        noisy_audio_paths = sorted(glob(self.input_audio_dir + '/**/*.wav', recursive=True))
        self.file_paths = [paths[len(self.input_audio_dir):] for paths in noisy_audio_paths]
        # use label binarizer for one hot encoding for phoneme labels
        self.lb = LabelBinarizer().fit(phoneme_set)
        self.lbViseme = LabelBinarizer().fit(viseme_set)
        
        self.dataset_len = len(self.file_paths)

    # z-normalization for audio data (subtract mean and devide by variance)
    def normalisation(self, i):
        i_std = np.std(i)
        if i_std == 0.:
            i_std = 1.
        return (i - np.mean(i))/i_std

    def __getitem__(self, i):
        
        if self.audio or self.mfcc:
            # choose random noise_type and SNR
            snr = self.snrs[random.randint(0, len(self.snrs)-1)]

            test_speech = librosa.load(self.audio_dir + snr + '/' + self.file_paths[i], sr=16000)[0]
            audio = self.normalisation(test_speech)

        if self.mfcc: 

            # use librosa mfcc and delta calculation to obtain the mfccs and their derivatives
            mfccs = mfcc(audio, sr=16000, hop_length=256, n_mfcc=13, win_length=1024)
            mfccs_v = delta(mfccs, 5)
            mfccs_acc = delta(mfccs_v, 5)

            mfccs_complete = np.transpose(np.concatenate((mfccs, mfccs_v, mfccs_acc)))
            #print(mfccs_complete.shape)

        if self.video:
            video = np.load(self.video_dir +self.file_paths[i][:-4] + '.npy')
            video = np.moveaxis(video, -1, 0)
            
        
        # load labels with a frame rate of 30 frames/s
        labels = np.load(self.labels_dir + self.file_paths[i][:-4] + '.npy')
        labels = np.array([n.encode("ascii", "ignore") for n in labels])
                

        if self.video and not (self.audio or self.mfcc):
            labels = [phone_vis_trans[str(label)[2:-1]] for label in labels]
            labels = self.lbViseme.transform(labels)         
        else:
            labels = self.lb.transform(labels)

        if self.audio:
            if(len(audio)/256 < len(labels)):
                labels=labels[:int(len(audio)/256)]
            elif(len(audio)/256 > len(labels)):
                audio=audio[:len(labels)*256]

        if self.mfcc:
            if mfccs_complete.shape[0] > labels.shape[0]:
                mfccs_complete = mfccs_complete[:labels.shape[0], :]
            elif mfccs_complete.shape[0] < labels.shape[0]:
                labels = labels[:mfccs_complete.shape[0], :]

        if self.video:
            if video.shape[0] > labels.shape[0]:
                video = video[:labels.shape[0], :, :]

            elif video.shape[0] < labels.shape[0]:
                labels = labels[:video.shape[0], :]
                if self.mfcc:
                    mfccs_complete = mfccs_complete[:video.shape[0], :]
                if self.audio:
                    audio=audio[:len(labels)*256]

        if self.video:
            if self.audio:
                if self.mfcc:
                    return torch.from_numpy(audio), torch.from_numpy(video), torch.from_numpy(mfccs_complete), torch.from_numpy(labels)
                else:
                    return torch.from_numpy(audio), torch.from_numpy(video), torch.from_numpy(labels), self.file_paths[i]
            elif self.mfcc: 
                return torch.from_numpy(video), torch.from_numpy(mfccs_complete), torch.from_numpy(labels)    
            else:
                return torch.from_numpy(video), torch.from_numpy(labels)   
        elif self.audio:
            return torch.from_numpy(audio), torch.from_numpy(labels)       
        elif self.mfcc:
            return torch.from_numpy(mfccs_complete[:,:512]), torch.from_numpy(labels)

        
    def __len__(self):
        return self.dataset_len

# collate function for audio
def pad_collate_audio(batch):
    video_shape = 67
    mfccs = False
    videos = False
    if len(batch[0]) == 4:
        (audio, video, mfcc, labels) = zip(*batch)
        mfccs = True
        videos = True
    elif len(batch[0]) == 3:
        (audio, video, labels) = zip(*batch)
        videos = True
    else: 
        (audio, labels) = zip(*batch)
    audio_len = [len(a) for a in audio]
    label_len = [len(y) for y in labels]
    
    if min(audio_len)%2 != 0:
        print("Something went wrong with audio")

    a_len = min(audio_len)
    l_len = min(label_len)
    audio_short = torch.zeros(len(audio), a_len)
    if videos:
        video_short = torch.zeros(len(video), l_len, video_shape, video_shape)
    label_short = torch.zeros(len(labels), l_len, len(phoneme_set))
    if mfccs:
        mfcc_short = torch.zeros(len(mfcc), l_len, 39)

    for i in range(len(audio)):
        temp_len_a = int((len(audio[i])-a_len)/2)
        temp_len_l = int((len(labels[i])-l_len)/2)

        if temp_len_l > 0:
            if len(labels[i])-l_len - temp_len_l == temp_len_l:
                label_short[i] = labels[i][temp_len_l:-temp_len_l, :]
                if mfccs:
                    mfcc_short[i]  = mfcc[i][temp_len_l:-temp_len_l, :]
                if videos:
                    video_short[i] = video[i][temp_len_l:-temp_len_l, :, :]
            else: 
                label_short[i] = labels[i][temp_len_l+1:-temp_len_l, :]
                if mfccs:
                    mfcc_short[i]  = mfcc[i][temp_len_l+1:-temp_len_l, :]
                if videos:
                    video_short[i] = video[i][temp_len_l+1:-temp_len_l, :, :]
        else: 
            label_short[i] = labels[i][:l_len,:]
            if mfccs:
                mfcc_short[i] = mfcc[i][:l_len, :]
            if videos:
                video_short[i] = video[i][:l_len, :, :]
    
        if temp_len_a > 0:
            audio_short[i] = audio[i][temp_len_a:-temp_len_a]
        else:
            audio_short[i] = audio[i]

    if len(batch[0]) == 4:
        return audio_short, video_short, mfcc_short, label_short
    elif len(batch[0]) == 3:
        return audio_short, video_short, label_short
    else:
        return audio_short, label_short
    
# collate function for video
def pad_collate_video(batch):
    video_shape = 67
    mfccs = False
    if len(batch[0]) == 3:
        (video, mfcc, labels) = zip(*batch)
        label_len = [len(y) for y in labels]

        l_len = min(label_len)
        label_short = torch.zeros(len(labels), l_len, len(phoneme_set))
        
        mfccs = True
    else: 
        (video, labels) = zip(*batch)
        label_len = [len(y) for y in labels]
        l_len = min(label_len)
        label_short = torch.zeros(len(labels), l_len, len(viseme_set))

    video_short = torch.zeros(len(video), l_len, video_shape, video_shape)
    
    if mfccs:
        mfcc_short = torch.zeros(len(mfcc), l_len, 39)

    for i in range(len(video)):
        temp_len_l = int((len(labels[i])-l_len)/2)

        if temp_len_l > 0:
            if len(labels[i])-l_len - temp_len_l == temp_len_l:
                label_short[i] = labels[i][temp_len_l:-temp_len_l, :]
                if mfccs:
                    mfcc_short[i]  = mfcc[i][temp_len_l:-temp_len_l, :]
                video_short[i] = video[i][temp_len_l:-temp_len_l, :, :]
            else: 
                label_short[i] = labels[i][temp_len_l+1:-temp_len_l, :]
                if mfccs:
                    mfcc_short[i]  = mfcc[i][temp_len_l+1:-temp_len_l, :]
                video_short[i] = video[i][temp_len_l+1:-temp_len_l, :, :]
        else: 
            label_short[i] = labels[i][:l_len,:]
            if mfccs:
                mfcc_short[i] = mfcc[i][:l_len, :]
            video_short[i] = video[i][:l_len, :, :]
    if len(batch[0]) == 3:
        return video_short, mfcc_short, label_short
    else:
        return video_short, label_short

#collate function for mfcc
def pad_collate_mfcc(batch):
    (mfcc, labels) = zip(*batch)
    label_len = [len(y) for y in labels]

    l_len = min(label_len)

    label_short = torch.zeros(len(labels), l_len, len(phoneme_set))
    mfcc_short = torch.zeros(len(mfcc), l_len, 39)

    for i in range(len(mfcc)):
        temp_len_l = int((len(labels[i])-l_len)/2)

        if temp_len_l > 0:
            if len(labels[i])-l_len - temp_len_l == temp_len_l:
                label_short[i] = labels[i][temp_len_l:-temp_len_l, :]
                mfcc_short[i]  = mfcc[i][temp_len_l:-temp_len_l, :]
            else: 
                label_short[i] = labels[i][temp_len_l+1:-temp_len_l, :]
                mfcc_short[i]  = mfcc[i][temp_len_l+1:-temp_len_l, :]
        else: 
            label_short[i] = labels[i][:l_len,:]
            mfcc_short[i] = mfcc[i][:l_len, :]

    return mfcc_short, label_short
