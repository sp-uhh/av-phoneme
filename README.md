# Continous Phoneme Recognition based on Audio-Visual Modality Fusion

## Abstract

While state-of-the-art audio-only phoneme recognition is already at a high standard, the robustness of existing methods still drops in very noisy environments. To mitigate these limitations, visual information can be incorporated into the recognition system, such that the problem is formulated in a multi-modal setting. To this end, we develop a continuous, audio-visual phoneme classifier that takes raw audio waveforms and video frames as input. Both modalities are processed by individual feature extraction models before a fusion model exploits their correlations. Audio features are extracted with a residual neural network, while video features are obtained with a convolutional neural network. Furthermore, we model temporal dependencies with gated recurrent units. For modality fusion, we compare simple concatenation, attention-based methods, as well as squeeze-and-excitation to learn a joint representation. We train our models on the NTCD-TIMIT dataset, using distinct noise types from the QUT dataset for the test. By pre-training the feature extraction models on the individual modalities first, we achieve best performance for the audio-visual model that is trained end-to-end. In the experiments, we show that by including the video modality, we increase the accuracy of phoneme prediction by 9\% in very noisy acoustic environments. The results indicate that in such environments our approach remains more robust compared to existing methods. The code and pre-trained models are available.

## Settings

To run the scripts change dir_input, dir_audio, dir_video, dir_labels args to local paths.

Pre-traind models can be downloaded here: 
https://fbicloud.informatik.uni-hamburg.de/index.php/s/sbSLBaJepzsKzYW

Whenever you use this code for any experiments and/or publications you need to cite our original paper.
