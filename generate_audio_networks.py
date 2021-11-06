# %%

import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
import os
import numpy as np

import random
def train_spec_and_convert(spec_model, index, pathDetails, training_dataset, binary_output):
    # first get the size prefix of this model's name:
    
    # const_prefix = "spec_g_"
    # size_prefix = str(spec_model.layers[2].filters) + "-" + str(spec_model.layers[3].filters) + "-" + str(spec_model.layers[7].units) + "_"
    # perf_prefix = ""
    
    # first train the given model
    print("Training Model")
    trainedModel, accuracy, loss = train_model(spec_model, training_dataset, binary_output)
    lossStr = None
    accuracyStr = None
    try:
        lossStr = round(loss*100)
    except:
        lossStr = 1

    try:
        accuracyStr = round(accuracy*100)
    except:
        accuracyStr = 0
    
    perf_prefix = str(lossStr) + "-" + str(accuracyStr)
    # identifier = const_prefix + str(index)  + "_"+ size_prefix + perf_prefix
    fullModelName = pathDetails[2] + "_" + str(index) + "_" + perf_prefix
    modelPath = os.path.join(pathDetails[0], fullModelName)
    trainedModel.save(modelPath) # save model
    print("Saved Model")
    fullTfliteName = pathDetails[2] + "_"  + str(index) + "_" + perf_prefix + pathDetails[3]
    tflitePath = os.path.join(pathDetails[1], fullTfliteName)
    # now to tack on the 4d reshape to allow for conv compatibility
    global existing_tflite_model
    existing_tflite_model = trainedModel

    hybrid_model = CNNOp()
    print("Creating tflite Model")
    # convert to tflite model
    convert_to_tflite_model(hybrid_model, tflitePath)

    print("converting to cpp flatbuffer")
    # generate C file in wsl
    path = "/mnt/c/Users/Aeyohan/Documents/work/ENGG4811/code/Tensorflow/"
    linux_tflite = to_linux_compat_path(tflitePath)
    linux_flatbuffer = linux_tflite.split(".tflite")[0] + ".cpp"
    command = 'ubuntu run "xxd -i ' + path + linux_tflite + ' > ' + path + linux_flatbuffer + '"' 
    os.system(command)
    # print("run command: " + command)
    return (trainedModel, hybrid_model, modelPath, tflitePath, (loss, accuracy))

    # then convert it to a tensorflow

def to_linux_compat_path(path):
    return "/".join(path.split("\\")) # changes \\ from windows to / in linux


def train_model(model, dataSet, binary_output):
    
    trainSet = dataSet[0]
    testSet = dataSet[1]
    valSet = dataSet[2]


    # size = [None]
    # size.extend(stfts_fl16[0].shape.as_list())
    # input_shape = tf.TensorShape(size)
    # input_shape = graphs_int16[0].shape
    # norm_layer = preprocessing.Normalization()
    # # norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))
    # num_labels = len(raininglabelList)
    # num_windows = 5

    

    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    EPOCHS = 20
    history = model.fit(
        trainSet, 
        validation_data=valSet,  
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )
    loss, accuracy = model.evaluate(testSet)
    print("Loss:", loss)
    print("Accuracy:", accuracy)

    return (model, accuracy, loss)
existing_tflite_model = None

class CNNOp(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[32,32], dtype=tf.int16)])
    def __call__(self,x):
        s1 = tf.expand_dims(x,2)
        s1 = tf.expand_dims(s1,0)
        s1 = tf.cast(s1, dtype=tf.float32)
        return existing_tflite_model(s1)

def generate_range_models(iterations=5, binary_output=True):

    completed = []
    desc = "binary" if binary_output else "type"
    compCacheName = "completed_" + desc + "_models.pic"
    if os.path.exists(compCacheName):
        # load pickle into completed
        print("Loading Cache file ")
        completedCache = open(compCacheName, 'rb')
        completed += pickle.load(completedCache)
        completedCache.close()
    else:
        print("Creating new cache file")
        completedCache = open(compCacheName, 'wb')
        pickle.dump(completed, completedCache)
        completedCache.close()
    # layer1_sizes = [8, 12, 16, 24, 32]
    # layer2_sizes = [12, 16, 24, 32, 64, 80, 96]
    # layer3_sizes = [12, 16, 24, 32, 64, 80, 96, 128, 160]

    layer1_sizes = [8, 12, 16, 20, 24, 32, 48]
    layer2_sizes = [12, 16, 24, 32, 48, 64, 80, 96]
    layer3_sizes = [12, 16, 24, 32, 48, 64, 80, 96, 128, 160]
    

    configs = []
    # newConfigs = []
    for config1 in layer1_sizes:
        for config2 in layer2_sizes:
            for config3 in layer3_sizes:
                # first check that layer 2 > layer 1 and that layer 3 > 0.5* layer2
                if (config2 < config1) or (config3 < int(0.5*config2)):
                    continue
                configs.append((config1, config2, config3))

    toBeCompleted = []
    for config in configs:
        if config not in completed:
            toBeCompleted.append(config)
    configs = toBeCompleted

    # configs = configs[-69:]

    print("Training " + str(len(configs)) + " network configurations")

    training_dataset = load_trimmed_dataset(binary_output)

    config_strings = ["-".join([str(config[0]), str(config[1]), str(config[2])]) for config in configs]
    # results = []

    colNames = ["Conv1", "Conv2", "Dense", "Loss", "Accuracy", "Model Params", "tflite Size"]
    for i in range(iterations):
        colNames.append("Loss_" + str(i+1))
    for i in range(iterations):
        colNames.append("Accuracy_" + str(i+1))

    csvOutName = desc + ' training data.csv'
    
    
    output = []
    if os.path.exists(csvOutName):
        temp = pd.read_csv(csvOutName)
        output = temp.values.tolist()
    status = 0
    str_i = 0
    for config in configs:
        result = train_networks_n_times(config, binary_output, training_dataset, n=iterations)
        # results.append(result)
        print("completed "+ str(status) + " of " + str(len(configs)-1) + " networks")
        status += 1
        completed.append(config)
        
        completedCache = open(compCacheName, 'wb')
        pickle.dump(completed, completedCache)
        completedCache.close()

        values = []
        configStr = config_strings[str_i].split("-")
        str_i += 1
        avgLoss = result[0][0]
        avgAcc = result[0][1]
        paramSize = result[iterations + 1][0][0]
        liteSize = result[iterations + 1][0][1] / 1000
        values = values + configStr
        values.append(avgLoss)
        values.append(avgAcc)
        values.append(paramSize)
        values.append(liteSize)
        for i in range(iterations):
            values.append(result[i][0])
        for i in range(iterations):
            values.append(result[i][1])
        # output.append([configStr, avgLoss, avgAcc, paramSize, liteSize, results[i][1][0], results[i][2][0], results[i][3][0], results[i][4][0], results[i][5][0], results[i][1][1], results[i][2][1], results[i][3][1], results[i][4][1], results[i][5][1]])
        output.append(values)
    
        output_df = pd.DataFrame(output, columns=colNames)
        output_df.to_csv(csvOutName, index=False) # prevent row nums

    print("Writing details to CSV")
    # write results to csv?
    # first create a dataframe
    
    for i in range(len(configs)):
        pass

def train_networks_n_times(config, binary_output, training_dataset, n=5):
    modelStats = []
    perfResults = [] # 0 contains average, 1-5 contains element wise performance metrics
    # create a model for given config
    pathDetails = generate_fileNames(config, binary_output)

    model = get_custom_model(config, binary_output)
    models = []
    for i in range(n):
        trainedModel, tfliteModel, tmPath, tfPath, perf = train_spec_and_convert(model, i, pathDetails, training_dataset, binary_output)
        # get the required metrics: total paramters, file size, loss, accuracy
        perfResults.append(perf)
        if i == 0:
            params = trainedModel.count_params()
            file_size = os.path.getsize(tfPath)
            modelStats.append((params, file_size))

    # also get agregated results
    lossTotal = 0
    accuracyTotal = 0
    for loss, accuracy in perfResults:
        lossTotal += loss
        accuracyTotal += accuracy
    loss = lossTotal / n
    accuracy = accuracyTotal / n
    perfResults.insert(0, (loss, accuracy))
    perfResults.append(modelStats)
    return (perfResults)

def generate_fileNames(config, binary_output):
    desc = "binary" if binary_output else "type"
    config_name = str(config[0]) + "_" + str(config[1]) + "_" + str(config[2])
    prefix = "spec_g_" + config_name # after this goes index and then performance
    suffix = ".tflite"
    if not os.path.exists(desc + "_models"):
        os.mkdir(desc + "_models")
    model_path = os.path.join(desc + "_models", prefix)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(desc + "_lite_models"):
        os.mkdir(desc + "_lite_models")
    tflite_path = os.path.join(desc + "_lite_models", prefix)
    if not os.path.exists(tflite_path):
        os.mkdir(tflite_path)

    return (model_path, tflite_path, prefix, suffix)

def load_trimmed_dataset(binary):
    
    
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

    # # split between yes and no
    # rainLabels = [rainLabels.loc[rainLabels == value] for value in raininglabelList]
    # typeLabels = [typeLabels.loc[typeLabels == value] for value in typelabelList]
    
    # rainLabels = [label.tolist() for label in rainLabels]
    # typeLabels = [label.tolist() for label in typeLabels]

    # # sample the lists to even out training
    # # find the second largest set
    # rainLens = [len(label) for label in rainLabels]
    # typeLens = [len(label) for label in typeLabels]

    # rainLens.remove(max(rainLens))
    # typeLens.remove(max(typeLens))

    # # create the upper limit of other data as 1.5x the samples
    # # i.e. for the 100 rain samples, there can be 150 non-rain
    # rainLimit = round(max(rainLens) * 1.5)
    # typeLimit = round(max(typeLens) * 2.5) # allow 2.5 to identify None as priority

    # # sample data from each label
    # temp = []
    # for label in rainLabels:
    #     if len(label) > rainLimit:
    #         temp += random.sample(label, rainLimit)
    #     else:
    #         temp += label
    # rainLabels = temp
    # label = []
    # for label in typeLabels:
    #     if len(label) > typeLimit:
    #         temp += random.sample(label, typeLimit)
    #     else:
    #         temp += label
    # typeLabels = temp

    # rainLabelsSliced = []
    # typeLabelsSliced = []
    
    # for value in rainLabels:
    #     rainLabelsSliced.extend([value,value,value,value,value])

    # for value in typeLabels:
    #     typeLabelsSliced.extend([value,value,value,value,value])
    rainLabels = rainLabels.tolist()
    typeLabels = typeLabels.tolist()

    rainLabelsSliced = []
    typeLabelsSliced = []

    for value in rainLabels:
        rainLabelsSliced.extend([value,value,value,value,value])

    for value in typeLabels:
        typeLabelsSliced.extend([value,value,value,value,value]) 

    selData = None
    if binary:
        selData = rainLabelsSliced
    else:
        selData = typeLabelsSliced

    # typeLabelsSliced = tf.convert_to_tensor(typeLabelsSliced)
    # preprocess stfts as necesasry
    ft_ds = tf.data.Dataset.from_tensor_slices((graphs_int16, selData))
    ft_ds.element_spec

    cachedSet = ft_ds.cache()
    cachedSet = cachedSet.shuffle(8000)

    setSize = len(selData) # should be 1413
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

    return [trainSet, testSet, valSet]

# binary output specifies if the outputs should specify if it's raining or how hard
def get_custom_model(config, binary_output):
    layer1 = config[0]
    layer2 = config[1]
    layer3 = config[2]
    k1_size = (2 if layer1 < 16 else (3 if layer1 <= 24 else 4))
    k2_size = (3 if layer2 <= 16 else (4 if layer1 <= 64 else 5))
    model = models.Sequential([
        layers.InputLayer(input_shape=tf.TensorShape([32,32,1])),
        preprocessing.Resizing(layer1, layer1), 
        preprocessing.Normalization(),
        layers.Conv2D(layer1, k1_size, activation=tf.nn.relu),
        layers.Conv2D(layer2, k2_size, activation=tf.nn.relu),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(layer3, activation=tf.nn.relu),
        layers.Dropout(0.5),
        layers.Dense((2 if binary_output else 6))
    ], name='stft_model')

    return model

# %%
def convert_to_tflite_model(model, fileName):
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    concrete_func = model.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
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
    return tflite_model

    

# %% 

# ha ha, ml model gen go brrr
if __name__ == "__main__":
    
    # generate_range_models(iterations=5) 
    generate_range_models(iterations=5, binary_output=False)

# %%
