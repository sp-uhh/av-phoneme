import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelBinarizer
from model_video import LipNet
from model_audio import speech_model, mfcc_model
from model_dual import AVGRUBase, AVGRUSE, AVEffAttGRU, LipNetAV, speech_modelAV 
from data_handling import NTCDTIMITTEST, phoneme_set, viseme_set
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

# parser to be able to submit settings while calling the script
parser = argparse.ArgumentParser(description='Input Parameters for Predicted Phonemes')
parser.add_argument('--audio', default=1, type=int, help='train with audio data, 0=False')
parser.add_argument('--video', default=1, type=int, help='train with video data, 0=False')
parser.add_argument('--mfcc', default=0, type=int, help='train with mfcc, 1=True')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs, default=50')
parser.add_argument('--save_path', default="./", help='path to save models, default= current directory')
parser.add_argument('--batch_size', default=16, type=int, help='batch size, default=16')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate, default=1e-3')
parser.add_argument('--name', default='Test', help='specify experiment name')
# DELETE PATH BEFORE SUBMISSION
parser.add_argument('--dir_input', default='/data/9liebold/NTCDTIMIT/clean_speech/test', type=dir_path, help='test data directory')
parser.add_argument('--dir_audio', default='/data/9liebold/Test_Data/QUT/', type=dir_path, help='test data directory')
parser.add_argument('--dir_video', default="/data/9liebold/NTCDTIMIT/upsampled_video/test/", type=dir_path, help='test data directory')
parser.add_argument('--dir_labels', default="/data/9liebold/NTCDTIMIT/labels/test", type=dir_path, help='test data directory')

args = parser.parse_args()

audioB = True
videoB = True
mfccB = False

if not args.audio and not args.video and not args.mfcc:
    print("Please choose one input type!")
    exit()
if not args.video:
    videoB = False
if not args.audio:
    audioB = False
if args.mfcc:
    mfccB  = True
if (not args.audio and not args.mfcc):
    print("Please choose one audio type!")
    exit()


# get model path
model_path_mfcc = './models/mfcc.pt'

# Create model and load parameters
if (audioB and videoB):
    model_video  = LipNetAV(viseme=12).to(device)
    model_speech = speech_modelAV(input_size=512, hidden_size=512, num_classes=len(phoneme_set)).to(device)
    model_path_concat = './models/fine-fune/AV-ft.pt'
    model_path_speech = './models/fine-fune/A-ft.pt'
    model_path_video = './models/fine-fune/V-ft.pt'
    #model_concat = AVEffAttGRU(input_size=1024, hidden_size=512, num_layers=2, num_classes=38, key_channels = 912, value_channels=456).to(device)
    model_concat = AVGRUBase(input_size=1536, hidden_size=512, num_layers=2, num_classes=38).to(device)
    #model_concat = AVGRUSE(input_size=1024, hidden_size=512, num_layers=2, num_classes=38, reduction=16).to(device)
    model_concat.load_state_dict(torch.load(model_path_concat))
    model_concat.eval()
    for param in model_concat.parameters(): param.requires_grad = False
else:   
    model_video  = LipNet(viseme=12).to(device)
    model_speech = speech_model(input_size=512, hidden_size=512, num_classes=len(phoneme_set)).to(device)
    model_path_video = './models/video_only.pt'
    model_path_speech = './models/audio_only.pt'
model_mfcc   = mfcc_model().to(device)

model_video.load_state_dict(torch.load(model_path_video))
model_speech.load_state_dict(torch.load(model_path_speech))

model_mfcc.load_state_dict(torch.load(model_path_mfcc))

model_video.eval()
model_speech.eval()

model_mfcc.eval()

# set parameters to False for testing
for param in model_video.parameters(): param.requires_grad = False
for param in model_speech.parameters(): param.requires_grad = False
for param in model_mfcc.parameters(): param.requires_grad = False

# One-hot encoding for labels
lb_viseme = LabelBinarizer()
lb_viseme.fit(viseme_set)
lb_phoneme = LabelBinarizer()
lb_phoneme.fit(phoneme_set)

snr = '20'
print(snr)
predicted = []

test_dataset = NTCDTIMITTEST(input_audio_dir=args.dir_input, audio_dir=args.dir_audio , video_dir=args.dir_video, labels_dir=args.dir_labels, audio = audioB, video = videoB, mfccs = mfccB, snrs = [snr]) 

# load only one utterance
#subset = torch.utils.data.Subset(test_dataset, [306])

# data loader for testing
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
    drop_last=True, num_workers=1, pin_memory=True)

# run model for each utterance
if (audioB and videoB):
    for batch_idx, (audio, video, labels_loader, file) in enumerate(tqdm(test_loader)):
        audio = audio.float()
        audio_inputs = audio.to(device, non_blocking=True)
        video = video.float()
        video_inputs = video.to(device, non_blocking=True)
        video = model_video(video_inputs)
        outputs = model_speech(audio_inputs)
        outputs = model_concat(outputs, video)  
        y_score = torch.squeeze(outputs).cpu()
        predicted.append(y_score)
elif audioB:
    for batch_idx, (audio, labels_loader) in enumerate(tqdm(test_loader)):
        audio = audio.float()
        audio_inputs = audio.to(device, non_blocking=True)
        outputs = model_speech(audio_inputs) 
        y_score = torch.squeeze(outputs).cpu()
        predicted.append(y_score)
elif mfccB: 
    for batch_idx, (mfcc, labels_loader) in enumerate(tqdm(test_loader)):
        mfcc = mfcc.float()
        mfcc_inputs = mfcc.to(device, non_blocking=True)
        outputs = model_mfcc(mfcc_inputs) 
        y_score = torch.squeeze(outputs).cpu()
        predicted.append(y_score)

# save predictions
a = torch.ones(1,len(phoneme_set))
for sublist in predicted:
    for i in range(len(sublist)):
        sublist[i] = sublist.cpu()[i]
    a = torch.cat((a, sublist), dim=0)
y_preds = a[1:,:]

# save phonemes in txt file
exp_name = args.name
phonemes_pred = np.argmax(y_preds,axis=-1)

# save phonemes indecies 
np.savetxt(exp_name+"_phonemes_idx.txt", phonemes_pred, fmt='%i',delimiter=",")

# save phonemes list as
phonemes_txt = []
for i in range(len(phonemes_pred)):
    phonemes_txt.append(sorted(phoneme_set)[phonemes_pred[i]])
with open(exp_name+"_phonemes_c.txt", "w") as output:
    output.write(str(phonemes_txt))
output.close()