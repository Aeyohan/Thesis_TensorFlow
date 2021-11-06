
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


# %%
model_list = [
    # "spec_g_8_12_12/spec_g_8_12_12_4_44-75",
    # "spec_g_8_24_12/spec_g_8_24_12_4_37-82",
    # "spec_g_8_32_32/spec_g_8_32_32_4_33-83",
    # "spec_g_8_48_24/spec_g_8_48_24_4_22-92",
    # "spec_g_16_16_24/spec_g_16_16_24_3_20-93",
    # "spec_g_8_48_80/spec_g_8_48_80_4_26-90",
    # "spec_g_20_24_16/spec_g_20_24_16_4_59-73",
    # "spec_g_8_96_64/spec_g_8_96_64_4_17-94",
    # "spec_g_20_24_32/spec_g_20_24_32_1_29-88",
    # "spec_g_16_32_48/spec_g_16_32_48_3_16-94",
    "spec_g_24_64_80/spec_g_24_64_80_1_8-97",
    # "spec_g_20_48_96/spec_g_20_48_96_4_15-96",
    # "spec_g_16_48_96/spec_g_16_48_96_3_11-96"
]

model_list = [
    "spec_g_16_64_128/spec_g_16_64_128_4_8-98"
]

name_list = [file.split("/")[1] for file in model_list]

path_to = "binary_models"

models = [tf.keras.models.load_model(os.path.join(path_to, file)) for file in model_list]
existing_tflite_model = None


def load_trimmed_dataset(binary, type=tf.float32):
    
    cacheFile = open("tf_spec_graph_cache.pic", 'rb')
    data = pickle.load(cacheFile)

    files = data[0]
    spec_graphs = data[1]
    graphs_dim = [tf.expand_dims(value, 2) for value in spec_graphs]
    graphs_dim = [tf.cast(value, tf.float32) for value in graphs_dim]
    graphs_dim = [(value-133) / 2.1 for value in graphs_dim]
    graphs_int16 = [tf.cast(value, type) for value in graphs_dim]
    
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

    rainLabelsSliced = [tf.cast(value,tf.int16) for value in rainLabelsSliced]
    typeLabelsSliced = [tf.cast(value,tf.int16) for value in typeLabelsSliced]

    selData = None
    if binary:
        selData = rainLabelsSliced
    else:
        selData = typeLabelsSliced
    # graphs_int16 = tf.convert_to_tensor(graphs_int16)
    # selData = tf.convert_to_tensor(selData)
    # selData = tf.reshape(selData, [len(rainLabelsSliced), 1,1,1])
    # typeLabelsSliced = tf.convert_to_tensor(typeLabelsSliced)
    # preprocess stfts as necesasry
    ft_ds = tf.data.Dataset.from_tensor_slices(graphs_int16)
    ft_ds.element_spec

    # cachedSet = ft_ds.cache()
    shuffle = ft_ds.shuffle(8000)
    return shuffle 
    

def load_full_dataset(binary, cast=tf.int16):
    
    
    cacheFile = open("tf_spec_graph_cache.pic", 'rb')
    data = pickle.load(cacheFile)

    files = data[0]
    spec_graphs = data[1]
    graphs_dim = [tf.expand_dims(value, 2) for value in spec_graphs] 

    graphs_int16 = [tf.cast(value, cast) for value in graphs_dim]
    
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

    trainSet = cachedSet.take(len(selData) + 1)
    
    trainSet = trainSet.cache().shuffle(8000).batch(128).prefetch(tf.data.AUTOTUNE)
    
    return trainSet

    
int_quant_save_to="binary_int_quant_models"
# create representative dataset
print("Loading dataset")
rep_ds = load_trimmed_dataset(True, tf.uint8) # trainset
print("Finshed loading dataset")
def representative_data_gen():
  for input_value in rep_ds.batch(1).take(500):
    # Model has only one input so each data point has one element.
    yield [input_value]

if not os.path.exists(int_quant_save_to):
    os.mkdir(int_quant_save_to)


kmodels = [tf.keras.models.load_model(os.path.join(path_to, file)) for file in model_list]
existing_tflite_model = None

class ReshapeOp(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[32,32], dtype=tf.float32)])
    def __call__(self,x):
        s1 = tf.expand_dims(x,2)
        s1 = tf.expand_dims(s1,0)
        return existing_tflite_model(s1)


class AltCNNOp(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[32,32], dtype=tf.uint8)])
    def __call__(self,x):
        s1 = tf.expand_dims(x,2)
        s1 = tf.expand_dims(s1,0)
        # s1 = tf.cast(s1, dtype=tf.float32)
        return existing_tflite_model(s1)

for model, model_path in zip(kmodels, model_list):
    existing_tflite_model = model
    hybrid_model = AltCNNOp()
    concrete_func = hybrid_model.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.representative_dataset = representative_data_gen
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # tf.lite.OpsSet.TFLITE_BUILTINS
    # converter.inference_input_type = tf.float32
    # converter.inference_output_type = tf.float32
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()
    destination = os.path.join(int_quant_save_to, model_path.split("/")[1] + ".tflite")
    with open(destination, 'wb') as f:
        f.write(tflite_quant_model)
    path = "/mnt/c/Users/Aeyohan/Documents/work/ENGG4811/code/Tensorflow/"
    command = 'ubuntu run "xxd -i ' + path + destination + ' > ' + path + destination.split(".tflite")[0]+".cpp" + '"' 
    command = command.replace("\\", "/")
    print(command)
    os.system(command)

# %%
# Prep for testing (helper functinos)
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def get_loss(labels, inferred):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    return loss_fn(labels, inferred).numpy()

def evaluate_lite_model(interpreter, dataset, gen_conf_matrix=False):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    labels = []
    inference_raw = []
    inference_label = []
    for batch in dataset:
        labels += batch[1].numpy().tolist()
        input_data = [tf.reshape(value, [32,32]) for value in batch[0]]
        output_data = []
        for sample in input_data:
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            output_data.append(interpreter.get_tensor(output_details[0]['index']))
        inference_raw += output_data

    # produce the inferred labels by checking the max index, i.e. index 0 is
    # Not raining, index 1 is raining
    for value in inference_raw:
        inferred = max(value[0]) # get highest value
        index = value[0].tolist().index(inferred)
        inference_label.append(index)

    correct = [True if label == infer else False for label, infer in zip(labels, inference_label)]
    accuracy = sum(correct) / len(correct)
    loss = get_loss(labels, inference_raw)

    if gen_conf_matrix:
        f, ax = plt.subplots(1,1)
        plt.rcParams["figure.figsize"] = (4,4)
        ConfusionMatrixDisplay.from_predictions(labels, inference_label, ax=ax)
        ax.axes.set_yticklabels(["False", "True"], rotation = 90, va="center")
    return [loss, accuracy]



# %%
# perform batch testing for selected sets

if __name__ == "__main__":
    model_list = [
        "spec_g_8_12_12/spec_g_8_12_12_4_44-75",
        "spec_g_8_24_12/spec_g_8_24_12_4_37-82",
        "spec_g_8_32_32/spec_g_8_32_32_4_33-83",
        "spec_g_8_48_24/spec_g_8_48_24_4_22-92",
        "spec_g_16_16_24/spec_g_16_16_24_3_20-93",
        "spec_g_8_48_80/spec_g_8_48_80_4_26-90",
        "spec_g_20_24_16/spec_g_20_24_16_4_59-73",
        "spec_g_8_96_64/spec_g_8_96_64_4_17-94",
        "spec_g_20_24_32/spec_g_20_24_32_1_29-88",
        "spec_g_16_32_48/spec_g_16_32_48_3_16-94",
        "spec_g_24_64_80/spec_g_24_64_80_1_8-97",
        "spec_g_20_48_96/spec_g_20_48_96_4_15-96",
        "spec_g_16_48_96/spec_g_16_48_96_3_11-96"
    ]


    name_list = [file.split("/")[1] for file in model_list]

    path_to = "binary_models"
    lite_path = "binary_lite_models"
    fl16_path = "binary_fl16_models"
    int_path = "binary_int_quant_models"

    try:
        baseline_models
    except:
        baseline_models = [tf.keras.models.load_model(os.path.join(path_to, file)) for file in model_list]
    
    try:
        tf_set
    except:
        tf_set = load_full_dataset(True)

    try:
        tf_float_set
    except:
        tf_float_set = load_full_dataset(True)

    # perform inference on baseline models.

    try:
        baseline
    except:
        baseline = []
        for model in baseline_models:
            baseline.append(model.evaluate(tf_set))

    try:
        tflite_models
    except:
        lite = []
        tflite_models = [tf.lite.Interpreter(os.path.join(lite_path, file + ".tflite")) for file in model_list]
        count = 0
        for interpreter in tflite_models:
            interpreter.allocate_tensors()
            lite.append(evaluate_lite_model(interpreter, tf_set, gen_conf_matrix=True))
            plt.savefig("audioProcessingGraphs\\conf_mat_" + str(count) + "-"+ model_list[count].split('/')[0][7:] +".png", bbox_inches='tight')
            count += 1
            plt.show()
    
    try:
        fp16_models
    except:
        fp16 = []
        fp16_models = [tf.lite.Interpreter(os.path.join(lite_path, file + ".tflite")) for file in model_list]
        for interpreter in fp16_models:
            interpreter.allocate_tensors()
            fp16.append(evaluate_lite_model(interpreter, tf_float_set))
            
    try:
        int8_full_models
    except:
        int8_full = []
        int8_full_models = [tf.lite.Interpreter(os.path.join(lite_path, file + ".tflite")) for file in model_list]
        for interpreter in int8_full_models:
            interpreter.allocate_tensors()
            int8_full.append(evaluate_lite_model(interpreter, tf_float_set))

    
    print("baseline")
    print(baseline)

    print("lite")
    print(lite)

    print("fp16")
    print(fp16)

    print("int")
    print(int8_full)

    merge = []

    for i in range(len(baseline)):
        element = [model_list[i]] + baseline[i] + lite[i] + fp16[i] + int8_full[i]
        merge.append(element)

    attr_list = ["Loss", "Accuracy"]
    model_type = ["Baseline", "Lite", "FP16", "QuInt8"]

    cols = ["Model"]
    for m_type in model_type:
        for attr in attr_list:
            cols.append(m_type + " " + attr)


    df_data = pd.DataFrame(data=merge, columns=cols)
    df_data.to_csv("lite+quantisation testing.csv", index=False)
    
        
    
    


# %%
# also do the baseline 
def get_baseline_stats(model, dataset, baseline):
    labels = []
    inference_raw = []
    inference_label = []
    for batch in dataset:
        labels += batch[1].numpy().tolist()
        batchValues = model(batch)
        values = []
        for value in batchValues:
            values.append(value)
        inference_raw += values

    # produce the inferred labels by checking the max index, i.e. index 0 is
    # Not raining, index 1 is raining
    for value in inference_raw:
        inferred = max(value.numpy()) # get highest value
        index = value.numpy().tolist().index(inferred)
        inference_label.append(index)

    correct = [True if label == infer else False for label, infer in zip(labels, inference_label)]
    accuracy = sum(correct) / len(correct)
    loss = get_loss(labels, inference_raw)
    
    f, ax = plt.subplots(1,1)
    plt.rcParams["figure.figsize"] = (4,4)
    ConfusionMatrixDisplay.from_predictions(labels, inference_label, ax=ax)
    ax.axes.set_yticklabels(["False", "True"], rotation = 90, va="center")
    return [loss, accuracy]

if __name__ == "__main__":
    result = []
    count = 0
    for model in baseline_models:
        print(count)
        result.append(get_baseline_stats(model, tf_set, baseline))
        plt.savefig("audioProcessingGraphs\\conf_mat_baseline" + str(count) + "-"+ model_list[count].split('/')[0][7:] +".png", bbox_inches='tight')
        count += 1
        plt.show()
    
    merge = []

    for i in range(len(baseline)):
        element = [model_list[i]] + result[i] + lite[i] + fp16[i] + int8_full[i]
        merge.append(element)

    attr_list = ["Loss", "Accuracy"]
    model_type = ["Baseline", "Lite", "FP16", "QuInt8"]

    cols = ["Model"]
    for m_type in model_type:
        for attr in attr_list:
            cols.append(m_type + " " + attr)


    df_data = pd.DataFrame(data=merge, columns=cols)
    # df_data.to_csv("lite+quantisation testing.csv", index=False)
# %%
