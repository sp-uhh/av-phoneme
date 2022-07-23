# use python get_features.py > features.txt to save your features
import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from model_dual import LipNetAV, speech_modelAV
from data_handling import NTCDTIMITTEST, phoneme_set

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def init_variables():
    # parser to be able to submit settings while calling the script
    parser = argparse.ArgumentParser(description='Input Parameters for audio and video features. Use "python get_features.py > features.txt" to save your features or use\
        it in another file via "from get_features import get_input_features". get_input_features returns the features as a list')
    parser.add_argument('--audio', default=1, type=int, help='train with audio data, 0=False, default=1')
    parser.add_argument('--video', default=0, type=int, help='train with video data, 1=True, default=0')
    parser.add_argument('--save_path', default="./", help='path to save models, default=current directory')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size, default=1')
    parser.add_argument('--name', default='Test', help='specify experiment name')
    parser.add_argument('--dir_input', default='', type=dir_path, help='input data directory')
    parser.add_argument('--dir_audio', default='', type=dir_path, help='input audio data directory')
    parser.add_argument('--dir_video', default="", type=dir_path, help='input video data directory')
    parser.add_argument('--dir_labels', default="", type=dir_path, help='input labels data directory')

    args = parser.parse_args()

    bAudio = True
    bVideo = True

    if not args.audio and not args.video:
        print("Please choose one input type!")
        exit()
    if not args.video:
        bVideo = False
    if not args.audio:
        bAudio = False
    
    return args, bAudio, bVideo

def dataset_prep():
    args, bAudio, bVideo = init_variables()
    snr = ['20']

    test_dataset = NTCDTIMITTEST(input_audio_dir=args.dir_input, audio_dir=args.dir_audio , video_dir=args.dir_video, labels_dir=args.dir_labels, audio = bAudio, video = bVideo, snrs = snr) 

    # load only one utterance
    subset = torch.utils.data.Subset(test_dataset, [306])

    # data loader for testing
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
        drop_last=True, num_workers=1, pin_memory=True, collate_fn=pad_collate)

    return test_loader

# zero padding
def pad_collate(batch):
    args, bAudio, bVideo = init_variables()
    if len(batch[0]) == 4:
        (audio, video, labels, file) = zip(*batch)
        audio_pad = pad_sequence(audio, batch_first=True, padding_value=0)
        video_pad = pad_sequence(video, batch_first=True, padding_value=0)
        return audio_pad, video_pad, labels, file

    elif bAudio:
        (audio, labels) = zip(*batch)
        audio_pad = pad_sequence(audio, batch_first=True, padding_value=0)
        return audio_pad, labels

    elif bVideo:
        (video, labels) = zip(*batch)
        video_pad = pad_sequence(video, batch_first=True, padding_value=0)
        return video_pad, labels

def load_video_model():
    model_video  = LipNetAV(viseme=12).to(get_device())
    model_path_video = './models/video_only.pt'
    model_video.load_state_dict(torch.load(model_path_video))
    model_video.eval()
    for param in model_video.parameters(): param.requires_grad = False
    return model_video

def get_video_features(video, model_video):
    video = video.float()
    video_inputs = video.to(get_device(), non_blocking=True)
    video = model_video(video_inputs)
    return video

def load_audio_model():
    model_audio  = speech_modelAV(input_size=512, hidden_size=512, num_classes=len(phoneme_set)).to(get_device())
    model_path_audio = './models/audio_only.pt'
    model_audio.load_state_dict(torch.load(model_path_audio))
    model_audio.eval()
    for param in model_audio.parameters(): param.requires_grad = False
    return model_audio

def get_audio_features(audio, model_audio):
    audio = audio.float()
    audio_inputs = audio.to(get_device(), non_blocking=True)
    audio = model_audio(audio_inputs)
    return audio

def get_input_features(bAudio, bVideo):
    args, bAudio, bVideo = init_variables()
    test_loader = dataset_prep()
    if bAudio:
        all_audio_features = [] 
        model_audio = load_audio_model()
    if bVideo: 
        all_video_features =[]
        model_video = load_video_model()
    if bAudio and bVideo:
        for batch_idx, (audio, video, labels_loader, file) in enumerate(tqdm(test_loader)):
            all_audio_features.append(get_audio_features(audio, model_audio))
            all_video_features.append(get_video_features(video, model_video))
        return all_audio_features, all_video_features
    elif bAudio:
        for batch_idx, (audio, labels_loader) in enumerate(tqdm(test_loader)):
            all_audio_features.append(get_audio_features(audio, model_audio))
        return all_audio_features
    elif bVideo: 
        for batch_idx, (video, labels_loader) in enumerate(tqdm(test_loader)):
            all_video_features.append(get_audio_features(video, model_video))
        return all_video_features


if __name__ == '__main__':
    # get features with audio=True and video=False
    print(get_input_features(True, False))
