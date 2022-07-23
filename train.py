import os
import time
import torch
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader
from model_audio import speech_model, mfcc_model
from model_video import LipNet
from model_dual import AVGRUBase, AVGRUSE, AVEffAttGRU
from data_handling import NTCDTIMIT, pad_collate_mfcc, pad_collate_video, pad_collate_audio, phoneme_set, viseme_set
############################ SETTINGS ##########################################
# Time stemp for model file
time_stamp = time.localtime() 
time_string = time.strftime("%Y-%m-%d_%H:%M:%S", time_stamp)

# parser to be able to submit settings while calling the script
parser = argparse.ArgumentParser(description='Input Parameters for Training')
parser.add_argument('--audio', default=1, type=int, help='train with audio data, 0=False')
parser.add_argument('--video', default=1, type=int, help='train with video data, 0=False')
parser.add_argument('--mfcc', default=0, type=int, help='train with mfcc, 1=True')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs, default=50')
parser.add_argument('--save_path', default="./", help='path to save models, default= current directory')
parser.add_argument('--batch_size', default=16, type=int, help='batch size, default=16')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate, default=1e-3')
parser.add_argument('--name', default='Test', help='specify experiment name')

args = parser.parse_args()

print(str(args.name) + '_{}'.format(time_string))

epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr

num_workers = 12
pin_memory = True
non_blocking = True

# choose mode for training, for audio/video choose both 
train_audio = True
train_video = True
train_mfcc  = False

if not args.audio and not args.video and not args.mfcc:
    print("Please choose one input type!")
if not args.video:
    train_video = False
if not args.audio:
    train_audio = False
if args.mfcc:
    train_mfcc  = True

save_path = args.save_path
filename = save_path+args.name +'.txt'
print(filename)
path_csv = save_path+args.name +'.csv'

path_audio_model = './models/audio_only.pt'
path_video_model = './models/video_only.pt'
path_mfcc_model = ''

# For retraining an existing model 
model_path = ''
start_epoch = 1

########################## CONFIGURATION #######################################

logger_name = "mylog"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(filename, mode='a')
fh.setLevel(logging.INFO)
logger.addHandler(fh)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

# Computing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Edit the directories to your paths 
audio_data_dir = "/data/NTCDTIMIT_phoneme/clean_speech/"
noise_data_dir = "/data/NTCDTIMIT_phoneme/noisy_speech/"
val_data_dir = "/data/NTCDTIMIT_phoneme/noisy_speech/dev"
video_data_dir = "/data/NTCDTIMIT_phoneme/upsampled_video/"
labels_data_dir = "/data/NTCDTIMIT_phoneme/labels/"

train_dataset = NTCDTIMIT(input_audio_dir = audio_data_dir, dataset_type = 'train', noise_dir = noise_data_dir, val_dir = val_data_dir, video_dir=video_data_dir, labels_dir = labels_data_dir,audio=train_audio, video=train_video, mfccs=train_mfcc)
dev_dataset = NTCDTIMIT(input_audio_dir = audio_data_dir, dataset_type = 'dev', noise_dir = noise_data_dir, val_dir = val_data_dir, video_dir=video_data_dir, labels_dir = labels_data_dir, val= True, audio=train_audio, video=train_video, mfccs=train_mfcc)

# Configure data loader
if train_audio:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
        drop_last=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=pad_collate_audio)
    valid_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
        drop_last=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=pad_collate_audio)
    model_audio = speech_model(input_size=512, hidden_size=512, num_classes=38).to(device)
    if train_video:
        model_video = LipNet().to(device)
        model_concat = AVGRUBase(input_size=1536, hidden_size=512, num_layers=2, num_classes=38).to(device)
        model_video.load_state_dict(torch.load(path_video_model))
        model_audio.load_state_dict(torch.load(path_audio_model))
        
        # Define optimizer for model (w/o discriminator)
        for param in model_audio.parameters():
            param.requires_grad = False
        for param in model_video.parameters():
            param.requires_grad = False
        for param in model_concat.parameters():
            param.requires_grad = True
        optimizer_model = torch.optim.Adam([{'params': model_audio.parameters()}, \
            {'params': model_video.parameters()},\
                 {'params': model_concat.parameters()}], lr=learning_rate)
    else:
        optimizer_model = torch.optim.Adam(model_audio.parameters(), lr=learning_rate)
        model_audio.train()

elif train_video:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
        drop_last=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=pad_collate_video)
    valid_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
        drop_last=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=pad_collate_video)
    model_video = LipNet().to(device)

    if train_mfcc:
        model_mfcc = mfcc_model(input_size=39, hidden_size=512, num_classes=38).to(device)
        model_concat = AVGRUBase(input_size=1536, hidden_size=512, num_classes=38).to(device)
        model_video.load_state_dict(torch.load(path_video_model))
        model_mfcc.load_state_dict(torch.load(path_mfcc_model))
        
        # Define optimizer for model (w/o discriminator)
        for param in model_mfcc.parameters():
            param.requires_grad = False
        for param in model_video.parameters():
            param.requires_grad = False
        for param in model_concat.parameters():
            param.requires_grad = True
        optimizer_model = torch.optim.Adam(model_concat.parameters(), lr=learning_rate)
    else:
        optimizer_model = torch.optim.Adam(model_video.parameters(), lr=learning_rate)

elif train_mfcc:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
        drop_last=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=pad_collate_mfcc)
    valid_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
        drop_last=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=pad_collate_mfcc)
    model_mfcc = mfcc_model(input_size=39, hidden_size=512, num_classes=38).to(device)
    optimizer_model = torch.optim.Adam(model_mfcc.parameters(), lr=learning_rate)

train_loss_vec =[]
valid_loss_vec = []
data =[] # to store training loss in csv file

# Loss function cross entropy
def cross_entropy_phone(y_hat, y, eps, weights=False):
    w = torch.ones(len(phoneme_set))
    # phoneme set in alphabetical order, sil is index 29
    w[29] = 0.01
    w = w.to('cuda')
    if weights:
        return -torch.mean(torch.sum(w*y*torch.log(y_hat + eps), dim=-1)/torch.sum(w))
    else:
        return -torch.mean(torch.sum(y*torch.log(y_hat + eps), dim=-1)/len(phoneme_set))

# reciprocal weights
def cross_entropy_viseme(y_hat, y, eps, weights=False):
    w = torch.ones(len(viseme_set))
    w[0] = 1/0.0183
    w[1] = 1/0.0782
    w[2] = 1/0.0359
    w[3] = 1/0.0044
    w[4] = 1/0.0065
    w[5] = 1/0.0212
    w[6] = 1/0.0029
    w[7] = 1/0.0621
    w[8] = 1/0.1892
    w[9] = 1/0.1015
    w[10] = 1/0.0346
    w[11] = 1/0.4453
    w = w.to('cuda')
    if weights:
        return -torch.mean(torch.sum(w*y*torch.log(y_hat + eps), dim=-1))/torch.sum(w)
    else:
        return -torch.mean(torch.sum(y*torch.log(y_hat + eps), dim=-1)/len(viseme_set))

# Train
def train_A(epoch):
    model_audio.train()
    if train_video:
        model_video.train()
        model_concat.train()
    train_loss = 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
        drop_last=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=pad_collate_audio)
    
    for batch_idx, data in enumerate(tqdm(train_loader)):
        if len(data) == 2:
            audio, labels = data
            audio = audio.float()
            audio_inputs = audio.to(device, non_blocking=True)
            targets = labels.to(device)
            outputs = model_audio(audio_inputs)
        elif len(data) == 3: 
            audio, video, labels = data
            video = video.float()
            video_inputs = video.to(device, non_blocking=True)
            audio = audio.float()
            audio_inputs = audio.to(device, non_blocking=True)
            targets = labels.to(device)
            audio_outputs = model_audio(audio_inputs)
            video_outputs = model_video(video_inputs)
            outputs = model_concat(audio_outputs, video_outputs)

        loss = cross_entropy_phone(outputs, targets, 1e-8)
        
        # Backward pass
        loss.backward()  
        optimizer_model.step()
        optimizer_model.zero_grad()

        # Save losses 
        train_loss += loss.item()

    return train_loss

def train_V():
    model_video.train()
    if train_mfcc:
        model_mfcc.train()
        model_concat.train()
    train_loss = 0
    for batch_idx, data in enumerate(tqdm(train_loader)):
        if len(data) == 2:
            video, labels = data
            video_inputs = video.to(device, non_blocking=True)
            targets = labels.to(device)
            outputs = model_video(video_inputs)
            loss = cross_entropy_viseme(outputs, targets, 1e-8)

        elif len(data) == 3: 
            video, mfccs, labels = data
            video = video.float()
            video_inputs = video.to(device, non_blocking=True)
            mfccs = mfccs.float()
            mfccs_inputs = mfccs.to(device, non_blocking=True)
            targets = labels.to(device)
            audio_outputs = model_mfcc(mfccs_inputs)
            video_outputs = model_video(video_inputs)
            inputs = torch.cat((audio_outputs, video_outputs), dim=2)
            outputs = model_concat(inputs)
            loss = cross_entropy_phone(outputs, targets, 1e-8)
        
        # Backward pass
        loss.backward()  
        optimizer_model.step()
        optimizer_model.zero_grad()

        # Save losses 
        train_loss += loss.item()

    return train_loss

def train_M():
    model_mfcc.train()

    train_loss = 0
    for batch_idx, data in enumerate(tqdm(train_loader)):
        mfccs, labels = data
        mfccs = mfccs.float()
        mfccs_inputs = mfccs.to(device, non_blocking=True)
        targets = labels.to(device)
        outputs = model_mfcc(mfccs_inputs)
        loss = cross_entropy_phone(outputs, targets, 1e-8)
       
        # Backward pass
        loss.backward()  
        optimizer_model.step()
        optimizer_model.zero_grad()

        # Save losses 
        train_loss += loss.item()

    return train_loss

# validation method
def validate_A():
    model_audio.eval()
    if train_video:
        model_video.eval()
        model_concat.eval()
    valid_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(valid_loader)):
            if len(data) == 2:
                audio, labels = data
                audio = audio.float()
                audio_inputs = audio.to(device, non_blocking=True)
                targets = labels.to(device)
                outputs = model_audio(audio_inputs)
            elif len(data) == 3: 
                audio, video, labels = data
                video = video.float()
                video_inputs = video.to(device, non_blocking=True)
                audio = audio.float()
                audio_inputs = audio.to(device, non_blocking=True)
                targets = labels.to(device)
                
                audio_outputs = model_audio(audio_inputs)
                video_outputs = model_video(video_inputs)
                outputs = model_concat(audio_outputs, video_outputs)
            loss = cross_entropy_phone(outputs, targets, 1e-8)

            # Save losses 
            valid_loss += loss.item()

    if len(data) == 2:
        return valid_loss, model_audio
    else:
        return valid_loss, model_concat, model_audio, model_video

def validate_V():
    model_video.train()
    if train_mfcc:
        model_mfcc.train()
        model_concat.train()
    valid_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(valid_loader)):
            if len(data) == 2:
                video, labels = data
                video_inputs = video.to(device, non_blocking=True)
                targets = labels.to(device)
                outputs = model_video(video_inputs)
                loss = cross_entropy_viseme(outputs, targets, 1e-8)

            elif len(data) == 3: 
                video, mfccs, labels = data
                video = video.float()
                video_inputs = video.to(device, non_blocking=True)
                mfccs = mfccs.float()
                mfccs_inputs = mfccs.to(device, non_blocking=True)
                targets = labels.to(device)
                audio_outputs = model_mfcc(mfccs_inputs)
                video_outputs = model_video(video_inputs)
                inputs = torch.cat((audio_outputs, video_outputs), dim=2)
                outputs = model_concat(inputs)
                loss = cross_entropy_phone(outputs, targets, 1e-8)

            # Save losses 
            valid_loss += loss.item()

    if len(data) == 2:
        return valid_loss, model_video
    else:
        return valid_loss, model_concat

def validate_M():
    model_mfcc.train()
    valid_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(valid_loader)):
            mfccs, labels = data
            mfccs = mfccs.float()
            mfccs_inputs = mfccs.to(device, non_blocking=True)
            targets = labels.to(device)
            outputs = model_mfcc(mfccs_inputs)
            loss = cross_entropy_phone(outputs, targets, 1e-8)

            # Save losses 
            valid_loss += loss.item()

    return valid_loss, model_mfcc

# Start training
for epoch in range(start_epoch, epochs + 1):
    # Training step
    if train_audio:
        train_loss = train_A(epoch)
        valid_loss, model, modelA, modelV = validate_A()
    elif train_video:
        train_loss = train_V()
        valid_loss, model = validate_V()
    elif train_mfcc:
        train_loss = train_M()
        valid_loss, model = validate_M()
    else:
        print("Error")

    logger.info(train_loss)

    # Append train and validation loss for plotting
    data.append([train_loss/len(train_loader.dataset), valid_loss/len(valid_loader.dataset)])
    train_loss_vec.append(train_loss/len(train_loader.dataset))
    valid_loss_vec.append(valid_loss/len(valid_loader.dataset))

    df = pd.DataFrame(data, columns=['train loss', 'valid loss'])
    df.to_csv(path_csv)

    # Save best model
    if valid_loss_vec[-1] == min(valid_loss_vec):
        fileList = glob(save_path+str(args.name)+'AV_{}*.pt'.format(time_string))
        if fileList: os.remove(fileList[0])
        torch.save(model.state_dict(), save_path+str(args.name)+'AV_{}_{:03d}_vloss_{:.2f}.pt'.format(
            time_string, epoch, valid_loss / len(valid_loader.dataset)))
    print('- epoch: {}   train: {:.8f}   valid: {:.8f}'.format(
        epoch, train_loss/len(train_loader.dataset), 
        valid_loss/len(valid_loader.dataset)))
