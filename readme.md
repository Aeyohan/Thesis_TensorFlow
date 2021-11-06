# TensorFlow (Python)
This repository contains the python scripts used to create, test and evaluate
TensorFlow Lite models for audio and image based rain detection.

## Results
The results can be found in the .csv files for corresponding binary and type models
(Binary models only predict whether it is raining, while type models try to
predict the severity of rain).

TensorFlow Lite models have been included in the repo, but only from the
subset. The complete set of models consumes several GB's of space.


## Files
Files are python files with functionality separated into cells to use Jupyter
notebook to run section of files. Certain sections perform different things
that can be performed in isolation, e.g. load subset models, then convert to a
standard tflite, or convert to FP16 tflite, or convert to int8 or int16
quantised. These can be run independently as required.

Files should be run using Jupyter notebook, but most files should be fine being
called outright.
Some notable files include:
 - Files with an stft prefix are files for YAMNET's Spectrogram implementation.
    - or for the lite compatible spectrogram implementation from https://medium.com/@antonyharfield/converting-the-yamnet-audio-detection-model-for-tensorflow-lite-inference-43d049bd357c
 - Image_processing.py relates to loading image datasets and applying the temporal
   edge detection algorithm.
 - Generate_audio_networks.py: create network configurations and train them
 - tf_audio.py: load audio txt or wav files, convert them into TensorFlow audio
   and process them using YAMNET, YAMNET embeddings, spectrograms or
   lightweight spectrograms. 






