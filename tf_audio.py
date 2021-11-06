#%% Import required libraries
from functools import cache
import os
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object

from tensorflow.python.ops.gen_batch_ops import batch
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
import tensorflow_hub as hub
import tensorflow_io as tfio

import audiohelper
import stft_cnn
from queue import Queue
tf.config.list_physical_devices()
yamnet_batch = None
stft_batch = None
# Import the yamnet model
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)
#%% Utility functions for loading audio files and making sure the sample rate is correct.

# @tf.function
# def load_wav_16k_mono(filename):
#     """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
#     file_contents = tf.io.read_file(filename)
#     wav, sample_rate = tf.audio.decode_wav(
#           file_contents,
#           desired_channels=1)
#     wav = tf.squeeze(wav, axis=-1)
#     sample_rate = tf.cast(sample_rate, dtype=tf.int64)
#     wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
#     return wav

# # @tf.function
# # def load_wav_files(fileNameList):
# #     result = [load_wav_16k_mono(fileName) for fileName in fileNameList]
# #     return result

# def get_file_list(rootFolderPath):
#     # create an array of tupples, the tupple contains the path with file
#     # contents as another element
#     filePaths = []
#     for root, folders, files in os.walk(rootFolderPath):
#         for fileName in files:
#             if fileName.__contains__(".wav"):
#                 filePaths.append(os.path.join(root, fileName))
#     print("found " + str(len(files)) + " wav files")
#     # wavFiles = load_wav_files(filePaths)
    
#     return filePaths

# def load_batches_and_infer(folder):
#     print("fetching data from " + folder)
#     files = get_file_list(folder)
#     print("Loaded File metadata")
#     results = []
#     classResults = []
#     for i in range(len(files)):
#         file = files[i]
#         if (i % 10 == 0):
#             print("file " + str(i) + " of " + str(len(files)))
#         scores, embeddings, infClass, class_scores = get_yamnet_deduction(load_wav_16k_mono(file))
#         results.append(infClass)
#         classResults.append(class_scores)
#     print("Infered files")
#     # for i in range(len(files)):
#     #     print(files[i] + ": " + results[i])
#     return (files, results, classResults)
    
# @tf.function
# def get_yamnet_deduction(testData):
#     class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
#     class_names =list(pd.read_csv(class_map_path)['display_name'])

#     scores, embeddings, spectrogram = yamnet_model(testData)
#     class_scores = [tf.reduce_mean(score, axis=0) for score in scores]
#     top_classes = [tf.argmax(class_score) for class_score in class_scores]
#     inferred_classes = [class_names[top_class] for top_class in top_classes]
#     return(scores, embeddings, inferred_classes, class_scores)
#     # print(f'The embeddings shape: {embeddings.shape}')
    

#%% Utility functions for loading audio files and making sure the sample rate is correct.

@tf.function
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

# @tf.function
def load_wav_files(fileNameList):
    result = []
    for i in range(len(fileNameList)):
        if (i % 100 == 0):
            print("File " + str(i) + " of " + str(len(fileNameList)))
        fileName = fileNameList[i]
        result.append(load_wav_16k_mono(fileName))
    return result

def get_file_list(rootFolderPath):
    # create an array of tupples, the tupple contains the path with file
    # contents as another element
    filePaths = []
    for root, folders, files in os.walk(rootFolderPath):
        for fileName in files:
            if fileName.__contains__(".wav"):
                filePaths.append(os.path.join(root, fileName))
    print("found " + str(len(filePaths)) + " wav files")
    # wavFiles = load_wav_files(filePaths)
    
    return filePaths

def load_batches_and_yamnet_infer(folder):
    reloadData = False
    global yamnet_batch
    if yamnet_batch is None:
        reloadData = True
    elif yamnet_batch[0] != folder:
        reloadData = True
    files = None
    fileData = None

    if reloadData:
        print("fetching data from " + folder)
        files = get_file_list(folder)
        print("Loaded File metadata")
        fileData = load_wav_files(files)
        yamnet_batch = (folder, files, fileData)
        print("Finished loading file data")

    else: 
        files = yamnet_batch[1]
        fileData = yamnet_batch[2]
        print("Loaded cached Data")
    # results = []
    # classResults = []
    # for i in range(len(files)):
    #     # file = files[i]
    #     if (i % 10 == 0):
    #         print("file " + str(i) + " of " + str(len(files)))
    class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
    class_names =list(pd.read_csv(class_map_path)['display_name'])
    print("Performing inference",flush=True)
    scores, embeddings, top_classes = get_yamnet_deduction(fileData)
    inferred_class = [class_names[top_class] for top_class in top_classes]
        # results.append(infClass)
        # classResults.append(class_scores)
    print("Infered files")
    # for i in range(len(files)):
    #     print(files[i] + ": " + results[i])
    return (files, inferred_class, top_classes, embeddings)
    
@tf.function
def get_yamnet_deduction(testData):
    count = 0
    scoreSet = []
    embedSet = []
    topSet = []
    for data in testData:
        scores, embeddings, spectrogram = yamnet_model(data)
        scoreSet.append(scores)
        embedSet.append(embeddings)
        class_scores = tf.reduce_mean(scores, axis=0)
        top_class = tf.argmax(class_scores)
        topSet.append(top_class)
    return(scoreSet, embedSet, topSet)
    # print(f'The embeddings shape: {embeddings.shape}')

def load_batches_and_stft(folder):
    reloadData = False
    global stft_batch
    if stft_batch is None:
        reloadData = True
    elif stft_batch[0] != folder:
        reloadData = True
    files = None
    fileData = None

    if reloadData:
        print("fetching data from " + folder)
        files = get_file_list(folder)
        print("Loaded File metadata")
        fileData = load_wav_files(files)
        stft_batch = (folder, files, fileData)
        print("Finished loading file data")

    else: 
        files = stft_batch[1]
        fileData = stft_batch[2]
        print("Loaded cached Data")
    # preprocess data to restore to similar to raw input

    print("Performing inference",flush=True)
    # prepare inputs as a list
    rawData = fileData
    # fork out values 
    slicedData = get_batch_slice(rawData)
    slicedFiles = []
    for value in files:
        slicedFiles.extend([value, value, value, value, value])
    # scale
    slicedData = [element * 256 for element in slicedData]
    # slicedData = [tf.cast(entry, tf.float16) for entry in slicedData]
    # get results
    
    outputs = stft_cnn.get_batch_stft(slicedData)
    outputs = [tf.cast(entry, tf.float16) for entry in outputs]
        # results.append(infClass)
        # classResults.append(class_scores)
    print("Infered files")
    # for i in range(len(files)):
    #     print(files[i] + ": " + results[i])
    return (slicedFiles, outputs)

def load_batches_and_slice(folder):
    reloadData = False
    global stft_batch
    if stft_batch is None:
        reloadData = True
    elif stft_batch[0] != folder:
        reloadData = True
    files = None
    fileData = None

    if reloadData:
        print("fetching data from " + folder)
        files = get_file_list(folder)
        print("Loaded File metadata")
        fileData = load_wav_files(files)
        stft_batch = (folder, files, fileData)
        print("Finished loading file data")

    else: 
        files = stft_batch[1]
        fileData = stft_batch[2]
        print("Loaded cached Data")
    # preprocess data to restore to similar to raw input

    print("Performing slices",flush=True)
    # prepare inputs as a list
    rawData = fileData
    # fork out values 
    slicedData = get_batch_slice(rawData)
    slicedFiles = []
    for value in files:
        slicedFiles.extend([value, value, value, value, value])
    # scale
    slicedData = [element * 256 for element in slicedData] # rescale back up.
    # slicedData = [tf.cast(entry, tf.float16) for entry in slicedData]
    # get results
    return (slicedFiles, slicedData) 


@tf.function
def scale_up_256(samples):
    # training only & tfds
    samples = samples.map(lambda x: x * 256)
    tf.cast(samples, tf.float16)
    return samples

@tf.function
def get_batch_slice(samples):
    # training only
    batch = []
    for sample in samples:
        # Split 4s training sample into 5 x ~0.7s clips
        batch.extend(get_sample_slice(sample))
    return batch

@tf.function
def get_sample_slice(data):
    # wav only
    return (
        tf.slice(data, begin=[1], size=[10500]),
        tf.slice(data, begin=[11201], size=[10500]),
        tf.slice(data, begin=[22401], size=[10500]),
        tf.slice(data, begin=[33601], size=[10500]),
        tf.slice(data, begin=[44801], size=[10500])
    )

def get_sample_slice_raw(raw_data):
    # works with raw text value. Just don't forget to convert int to float at stage
    return (
        raw_data[1:10501],
        raw_data[11201:21701],
        raw_data[22401:32901],
        raw_data[33601:44101],
        raw_data[44801:55301]
    )

# %%
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models

def generate_stfts():
    suffix = "07 - Testing 2_0"
    files, stft = load_batches_and_stft("E:\\Data\\Thesis\\Audio\\filteredOutputs\\" + suffix)
    data = [files, stft]
    
    if os.path.exists("tf_stft_cache.pic"):
        # existing cahce fiel, load data and append to it
        cacheFile = open("tf_stft_cache.pic", 'rb')
        existingData = pickle.load(cacheFile)
        cacheFile.close()
        for i in range(len(existingData)):
            existingData[i].extend(data[i])
        cacheFile = open("tf_stft_cache.pic", 'wb')
        pickle.dump(existingData, cacheFile)
        cacheFile.close()
    else:
        # new file for new data
        cacheFile = open("tf_stft_cache.pic", 'wb')
        pickle.dump(data, cacheFile)
        cacheFile.close()
    
    #
    # data = {"File": files, "stft" }

def train_stfts():
    cacheFile = open("tf_stft_cache.pic", 'rb')
    data = pickle.load(cacheFile) 
    files = data[0]
    stfts = data[1]
    stfts_dim = [tf.expand_dims(value, 2) for value in stfts] 

    stfts_fl16 = [tf.cast(value, tf.float16) for value in stfts_dim]
    
    # load labels
    raininglabelList = {"Yes": 1, "No": 0}
    typelabelList = {"None": 0, "Very Light": 1, "Light": 2, "Moderate": 3, "Heavy": 4, "Very Heavy": 5}
    labelFile = pd.read_csv("E:\\Data\\Thesis\\Audio\\filteredOutputs\\labels.csv")
    folderList = labelFile["Folder"]
    sampleList = labelFile["Sample"]
    pathList = []
    for i in range(len(folderList)):
        pathList.append(folderList + "\\" + sampleList)
    rainLabels = labelFile["Raining"].apply(lambda rainLabel: raininglabelList[rainLabel])
    typeLabels = labelFile["Rain type"].apply(lambda typeLabel: typelabelList[typeLabel])

    rainLabels = rainLabels.tolist()
    typeLabels = typeLabels.tolist()

    rainLabelsSliced = []
    typeLabelsSliced = []

    for value in rainLabels:
        rainLabelsSliced.extend([value,value,value,value,value])

    for value in typeLabels:
        typeLabelsSliced.extend([value,value,value,value,value])
    # typeLabelsSliced = tf.convert_to_tensor(typeLabelsSliced)
    # preprocess stfts as necesasry
    ft_ds = tf.data.Dataset.from_tensor_slices((stfts_fl16, rainLabelsSliced))
    ft_ds.element_spec

    cachedSet = ft_ds.cache()
    cachedSet = cachedSet.shuffle(8000)

    setSize = len(rainLabelsSliced) # should be 1413
    trainSize = int(setSize * 0.8)
    testSize = int(setSize * 0.1)
    valSize = int(setSize * 0.1)

    trainSet = cachedSet.take(trainSize)
    testSet = cachedSet.skip(trainSize)
    valSet = testSet.skip(valSize)
    testSet = testSet.take(testSize)
    print("train:" + str(len(trainSet)), "test:" + str(len(testSet)), "val:" + str(len(valSet)) )
    trainSet = trainSet.cache().shuffle(8000).batch(32).prefetch(tf.data.AUTOTUNE)
    testSet = testSet.cache().batch(32).prefetch(tf.data.AUTOTUNE)
    valSet = valSet.cache().batch(32).prefetch(tf.data.AUTOTUNE)

    
    # size = [None]
    # size.extend(stfts_fl16[0].shape.as_list())
    # input_shape = tf.TensorShape(size)
    input_shape = stfts_fl16[0].shape
    norm_layer = preprocessing.Normalization()
    # norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))
    num_labels = len(raininglabelList)
    # num_windows = 5

    stft_model = models.Sequential([
        layers.Input(shape=input_shape),
        preprocessing.Resizing(16, 16), 
        preprocessing.Normalization(),
        layers.Conv2D(16, 2, activation=tf.nn.relu),
        layers.Conv2D(32, 3, activation=tf.nn.relu),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(32, activation=tf.nn.relu),
        layers.Dropout(0.5),
        layers.Dense(2)
    ], name='stft_model')

    stft_model.summary()
    stft_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    EPOCHS = 20
    history = stft_model.fit(
        trainSet, 
        validation_data=valSet,  
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )
    loss, accuracy = stft_model.evaluate(testSet)
    print("Loss:", loss)
    print("Accuracy:", accuracy)

    return stft_model
# %%
stft_model = train_stfts()
# %%
savePath = ".\\models\\stft_u"
stft_model.save(savePath, include_optimizer=False)
# %%
convert_to_tflite_model(stft_model, "stft_tflite.tflite")
# %%

def convert_to_tflite_model(model, fileName):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.inference_input_type = tf.uint16
    converter.target_spec.supported_types = [tf.uint16, tf.uint8, tf.uint32, tf.int32, tf.int16, tf.int8] 
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_output_type = tf.uint16   
    tflite_model = converter.convert()

    with open('.\\' + fileName, 'wb') as f:
        f.write(tflite_model)

# %%


def generate_embeddings():
    suffix = "07 - Testing 2_0"
    fileNames, classes, classResults, embeddings = load_batches_and_infer("E:\\Data\\Thesis\\Audio\\filteredOutputs\\" + suffix)
    data = [fileNames, classes, classResults, embeddings]
    import pickle
    if os.path.exists("tf_audio_cache.p"):
        # existing cahce fiel, load data and append to it
        cacheFile = open("tf_audio_cache.p", 'rb')
        existingData = pickle.load(cacheFile)
        cacheFile.close()
        for i in range(len(existingData)):
            existingData[i].extend(data[i])
        cacheFile = open("tf_audio_cache.p", 'wb')
        pickle.dump(existingData, cacheFile)
        cacheFile.close()
    else:
        # new file for new data
        cacheFile = open("tf_audio_cache.p", 'wb')
        pickle.dump(data, cacheFile)
        cacheFile.close()
    # for i in range(len(fileNames)):
    #     print(fileNames[i] + ": " + classes[i])
    data = {"File": fileNames, "Inferred class": classes}
    df = pd.DataFrame(data=data)
    df.to_csv("E:\\Data\\Thesis\\Audio\\filteredOutputs\\" + suffix + ".csv")
    print("Complete")
# # %%
# # load pickle data
# cacheFile = open("tf_audio_cache.p", 'rb')
# data = pickle.load(cacheFile)

# # Load data from CSV
# labels = pd.read_csv("E:\\Data\\Thesis\\Audio\\outputs\\labels.csv")
# full_path = 
# generate_embeddings()

# %%
import pickle
import pandas as pd
import tensorflow as tf

# def reshape(embeddings, rain, ty):
#     num = tf.shape(embeddings)[0]
#     temp = [embeddings[i][0] for i in range(len(embeddings))]
#     return (temp, tf.repeat(rain, num), tf.repeat(ty, num))

def train_embeddings():
    cacheFile = open("tf_audio_cache.p", 'rb')
    data = pickle.load(cacheFile)

    fileNames = data[0]
    embeddings = data[3]
    temp = [embeddings[i][0] for i in range(len(embeddings))]
    embeddings = temp
    raininglabelList = {"Yes": 1, "No": 0}
    typelabelList = {"None": 0, "Very Light": 1, "Light": 2, "Moderate": 3, "Heavy": 4, "Very Heavy": 5}
    labelFile = pd.read_csv("E:\\Data\\Thesis\\Audio\\filteredOutputs\\labels.csv")
    folderList = labelFile["Folder"]
    sampleList = labelFile["Sample"]
    pathList = []
    for i in range(len(folderList)):
        pathList.append(folderList + "\\" + sampleList)
    rainLabels = labelFile["Raining"].apply(lambda rainLabel: raininglabelList[rainLabel])
    typeLabels = labelFile["Rain type"].apply(lambda typeLabel: typelabelList[typeLabel])

    mainSet = tf.data.Dataset.from_tensor_slices((embeddings, rainLabels, typeLabels))
    mainSet.element_spec

    # mainSet.map(reshape).unbatch()
    mainSet.element_spec

    cachedSet = mainSet.cache()
    cachedSet = cachedSet.shuffle(2000)
    
    setSize = len(folderList) # should be 1413
    trainSize = int(setSize * 0.8)
    testSize = int(setSize * 0.1)
    valSize = int(setSize * 0.1)

    trainSet = cachedSet.take(trainSize)
    testSet = cachedSet.skip(trainSize)
    valSet = testSet.skip(valSize)
    testSet = testSet.take(testSize)

    trainSet = trainSet.cache().shuffle(1500).batch(32).prefetch(tf.data.AUTOTUNE)
    testSet = testSet.cache().batch(32).prefetch(tf.data.AUTOTUNE)
    valSet = valSet.cache().batch(32).prefetch(tf.data.AUTOTUNE)

    removeRain = lambda embeddings, rain, rainType: (embeddings, rainType)
    removeType = lambda embeddings, rain, rainType: (embeddings, rain)

    rainSet = (trainSet.map(removeType), testSet.map(removeType), valSet.map(removeType))
    typeSet = (trainSet.map(removeRain), testSet.map(removeRain), valSet.map(removeRain))

    rainModel = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                            name='rain_embedding'),
        tf.keras.layers.Dense(384, activation='relu'),
        tf.keras.layers.Dense(len(raininglabelList))
    ], name='rain_model')

    rainModel.summary()
    rainModel.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 optimizer="adam",
                 metrics=['accuracy'])
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=3,
                                            restore_best_weights=True)

    history = rainModel.fit(rainSet[0],
                       epochs=20,
                       validation_data=rainSet[2],
                       callbacks=callback)

    rainLoss, rainAccuracy = rainModel.evaluate(rainSet[1])
    

    typeModel = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                            name='type_embedding'),
        tf.keras.layers.Dense(384, activation='relu'),
        tf.keras.layers.Dense(len(typelabelList))
    ], name='type_model')

    
    typeModel.summary()
    typeModel.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 optimizer="adam",
                 metrics=['accuracy'])
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=3,
                                            restore_best_weights=True)
    history = typeModel.fit(typeSet[0],
                       epochs=20,
                       validation_data=typeSet[2],
                       callbacks=callback)
    
    typeLoss, typeAccuracy = typeModel.evaluate(typeSet[1])
    
    print("Rainset Loss: ", rainLoss)
    print("Rainset Accuracy: ", rainAccuracy)
    print("typeSet Loss: ", typeLoss)
    print("typeSet Accuracy: ", typeAccuracy)

    return (rainModel, typeModel)
# %%
models = train_embeddings()
rainModel = models[0]
typeModel = models[1]

savePath = ".\\models\\rainModel"
rainModel.save(savePath, include_optimizer=False)
savePath = ".\\models\\typeModel"
typeModel.save(savePath, include_optimizer=False)

# %%
testing_wav_file_name = tf.keras.utils.get_file('miaow_16k.wav',
                                                'https://storage.googleapis.com/audioset/miaow_16k.wav',
                                                cache_dir='./data/',
                                                cache_subdir='test_data')
@tf.function
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

testing_wav_data = load_wav_16k_mono(testing_wav_file_name)
yamnet_model(testing_wav_data)

yamnetConverter = tf.lite.TFLiteConverter.from_keras_model(yamnet_model)
yamnet_tflite_model = yamnetConverter.convert()

with open('.\\yamnet.tflite', 'wb') as f:
  f.write(yamnet_tflite_model)
# %%
rainConverter = tf.lite.TFLiteConverter.from_saved_model(".\\models\\rainModel")
rain_tflite_model = rainConverter.convert()

# Save the model.
with open('.\\rainModel.tflite', 'wb') as f:
  f.write(rain_tflite_model)
# %%
rainConverter = tf.lite.TFLiteConverter.from_saved_model(".\\models\\typeModel")
rain_tflite_model = rainConverter.convert()

# Save the model.
with open('.\\typeModel.tflite', 'wb') as f:
  f.write(rain_tflite_model)


# %%

import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models

def cache_audio_dir(folder):
    suffix = folder
    files, slices = load_batches_and_slice("E:\\Data\\Thesis\\Audio\\filteredOutputs\\" + suffix)
    data = [files, slices]
    
    if os.path.exists("tf_slice_cache.pic"):
        # existing cahce fiel, load data and append to it
        cacheFile = open("tf_slice_cache.pic", 'rb')
        existingData = pickle.load(cacheFile)
        cacheFile.close()
        for i in range(len(existingData)):
            existingData[i].extend(data[i])
        cacheFile = open("tf_slice_cache.pic", 'wb')
        pickle.dump(existingData, cacheFile)
        cacheFile.close()
    else:
        # new file for new data
        cacheFile = open("tf_slice_cache.pic", 'wb')
        pickle.dump(data, cacheFile)
        cacheFile.close()
# %%
cache_audio_dir("07 - Testing 2_0")

#  %%
import pickle
cacheFile = open("tf_slice_cache.pic", 'rb')
data = pickle.load(cacheFile) 
files = data[0]
slices = data[1]

# %%

class Process_stft(tf.Module):
  @tf.function(input_signature=[tf.TensorSpec(shape=[10500], dtype=tf.float32)])
  def __call__(self, x):
    return tf.square(x)

model = Squared()

concrete_func = model.__call__.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

# %%
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
def train_spec():
    cacheFile = open("tf_spec_graph_cache.pic", 'rb')
    data = pickle.load(cacheFile) 
    files = data[0]
    spec_graphs = data[1]
    graphs_dim = [tf.expand_dims(value, 2) for value in spec_graphs] 

    graphs_int16 = [tf.cast(value, tf.int16) for value in graphs_dim]
    
    # load labels
    raininglabelList = {"Yes": 1, "No": 0}
    typelabelList = {"None": 0, "Very Light": 1, "Light": 2, "Moderate": 3, "Heavy": 4, "Very Heavy": 5}
    labelFile = pd.read_csv("E:\\Data\\Thesis\\Audio\\filteredOutputs\\labels.csv")
    folderList = labelFile["Folder"]
    sampleList = labelFile["Sample"]
    pathList = []
    for i in range(len(folderList)):
        pathList.append(folderList + "\\" + sampleList)
    rainLabels = labelFile["Raining"].apply(lambda rainLabel: raininglabelList[rainLabel])
    typeLabels = labelFile["Rain type"].apply(lambda typeLabel: typelabelList[typeLabel])

    rainLabels = rainLabels.tolist()
    typeLabels = typeLabels.tolist()

    rainLabelsSliced = []
    typeLabelsSliced = []

    for value in rainLabels:
        rainLabelsSliced.extend([value,value,value,value,value])

    for value in typeLabels:
        typeLabelsSliced.extend([value,value,value,value,value])
    # typeLabelsSliced = tf.convert_to_tensor(typeLabelsSliced)
    # preprocess stfts as necesasry
    ft_ds = tf.data.Dataset.from_tensor_slices((graphs_int16, rainLabelsSliced))
    ft_ds.element_spec

    cachedSet = ft_ds.cache()
    cachedSet = cachedSet.shuffle(8000)

    setSize = len(rainLabelsSliced) # should be 1413
    trainSize = int(setSize * 0.8)
    testSize = int(setSize * 0.1)
    valSize = int(setSize * 0.1)

    trainSet = cachedSet.take(trainSize)
    testSet = cachedSet.skip(trainSize)
    valSet = testSet.skip(valSize)
    testSet = testSet.take(testSize)
    print("train:" + str(len(trainSet)), "test:" + str(len(testSet)), "val:" + str(len(valSet)) )
    trainSet = trainSet.cache().shuffle(8000).batch(32).prefetch(tf.data.AUTOTUNE)
    testSet = testSet.cache().batch(32).prefetch(tf.data.AUTOTUNE)
    valSet = valSet.cache().batch(32).prefetch(tf.data.AUTOTUNE)

    
    # size = [None]
    # size.extend(stfts_fl16[0].shape.as_list())
    # input_shape = tf.TensorShape(size)
    input_shape = graphs_int16[0].shape
    norm_layer = preprocessing.Normalization()
    # norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))
    num_labels = len(raininglabelList)
    # num_windows = 5

    spec_model = models.Sequential([
        layers.Input(shape=input_shape),
        preprocessing.Resizing(12, 12), 
        preprocessing.Normalization(),
        layers.Conv2D(12, 2, activation=tf.nn.relu),
        layers.Conv2D(16, 3, activation=tf.nn.relu),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(16, activation=tf.nn.relu),
        layers.Dropout(0.5),
        layers.Dense(2)
    ], name='stft_model')

    spec_model.summary()
    spec_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    EPOCHS = 20
    history = spec_model.fit(
        trainSet, 
        validation_data=valSet,  
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )
    loss, accuracy = spec_model.evaluate(testSet)
    print("Loss:", loss)
    print("Accuracy:", accuracy)

    return spec_model
# %%
spec_model = train_spec()
# %%
suffix = "spec_g_12-16-16_49-70"
savePath = ".\\models\\" + suffix
spec_model.save(savePath, include_optimizer=False)
# %%
convert_to_tflite_model(spec_model, ".\\lite_models\\" + suffix + ".tflite")
# %%

def convert_to_tflite_model(model, fileName):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    # converter.inference_input_type = tf.uint16
    converter.target_spec.supported_types = [tf.uint16, tf.uint8, tf.uint32, tf.int32, tf.int16, tf.int8] 
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_output_type = tf.uint16   
    tflite_model = converter.convert()

    with open('.\\' + fileName, 'wb') as f:
        f.write(tflite_model)

# %%