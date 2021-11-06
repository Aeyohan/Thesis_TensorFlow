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


from matplotlib import pyplot as plt

# %% Load dataset


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

# %% Create spectogram

if __name__ == "__main__":
    try: # check if train has already been loaded.
        train # if it doesn exists load it
        if train is None: # if it exists but is empty load it.
            train, test, val = load_trimmed_dataset(True)
    except:
        train, test, val = load_trimmed_dataset(True)
    

    raininglabelList = {"Yes": 1, "No": 0}
    values = [data for data in test]
    labels = values[0][1]
    samples = tf.split(values[0][0], num_or_size_splits=32, axis=0)

    raining_samples = []
    non_raining_samples = []
    for sample, label in zip(samples, labels):
        if label == 1:
            raining_samples.append(sample)
        else:
            non_raining_samples.append(sample)
    if not os.path.exists("audioProcessingGraphs"):
        os.mkdir("audioProcessingGraphs")

    seq = 0
    # raining sample spectograms
    print("Raining Samples")
    for sample in raining_samples:
        f, ax = plt.subplots(1, 1)    
        ax.imshow(tf.reshape(sample, [32,32]))
        ax.set_title("Raining Sample Spectogram")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        plt.savefig("audioProcessingGraphs\\spec_rain_" + str(seq) +".png", bbox_inches='tight')
        plt.show()
        seq += 1
    seq = 0
    # no raining spectograms
    print("Non-Raining Samples")
    for sample in non_raining_samples:    
        f, ax = plt.subplots(1, 1)  
        ax.imshow(tf.reshape(sample, [32,32]))
        ax.set_title("Non-Raining Sample Spectogram")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        plt.savefig("audioProcessingGraphs\\spec_non-rain_" + str(seq) +".png", bbox_inches='tight')
        plt.show()
        seq += 1
    
    f, ax = plt.subplots(1,4)  
    ax[0].imshow(tf.reshape(raining_samples[1], [32,32]))
    ax[0].set_title("Raining Sample")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Frequency")

    ax[1].imshow(tf.reshape(raining_samples[7], [32,32]))
    ax[1].set_title("Raining Sample")
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Frequency")

    ax[2].imshow(tf.reshape(non_raining_samples[2], [32,32]))
    ax[2].set_title("Non-Raining Sample")
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("Frequency")

    ax[3].imshow(tf.reshape(non_raining_samples[9], [32,32]))
    ax[3].set_title("Non-Raining Sample")
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[3].set_xlabel("Time")
    ax[3].set_ylabel("Frequency")

    plt.savefig("audioProcessingGraphs\\spec_combined.png", bbox_inches='tight')
    plt.show()

# %% Get confusion matrix for known data
from sklearn.metrics import ConfusionMatrixDisplay

def read_inference_log(folderPath):
    f = open(folderPath, 'r')
    f.readline()
    data = []
    for line in f:
        if not line.__contains__("Raining"):
            continue
        # data is available on this line
        text, non_rain, rain = line.split(",")
        data.append([text, int(non_rain), int(rain)])
    f.close()

    return data

# %% Initial testing
if __name__ == "__main__":
    sample1 = read_inference_log("audioProcessingGraphs\\LOG000.TXT")
    sample2 = read_inference_log("audioProcessingGraphs\\LOG003.TXT")
    sample3 = read_inference_log("audioProcessingGraphs\\LOG004.TXT")

    output1 = [True if value[0] == "Raining" else False for value in sample1]
    output2 = [True if value[0] == "Raining" else False for value in sample2]
    output3 = [True if value[0] == "Raining" else False for value in sample3]

    # clippings and trimmings (ignore start and end where opening and closing
    # the container result in loud noises simliar to rain shield  )
    
    output1 = output1[10:100]
    output2 = output2[5:75]
    # output 3 contains a period over 30 minutes where it started lightly
    # raining and stopped  & repeated preiodically. Splitting samples to reflect this
    output4 = output3[40:100] + output3[180:410]
    output3 = output3[5:35] + output3[100:150] # ignoring portion of light rain

    print("No rain sample. correct:" + str((len(output1) - sum(output1))/ len(output1)*100))
    plt.plot(output1)
    plt.show()
    print("Rain sample. correct:" + str(sum(output2)/ len(output2) * 100))
    plt.plot(output2)
    plt.show()
    print("No rain sample. correct:" + str((len(output3) - sum(output3))/ len(output3)*100))
    plt.plot(output3)    
    plt.show()
    print("Rain sample. correct:" + str(sum(output4)/ len(output4) * 100))
    plt.plot(output4)
    plt.show()

    raining_true = sum(output2) + sum(output4)
    raining_false = len(output2) - sum(output2) + len(output4) - sum(output4)
    non_raining_true = len(output1) - sum(output1) + len(output3) - sum(output3)
    non_raining_false = sum(output1) + sum(output3)
    print(raining_true)
    print(raining_false)
    print(non_raining_true)
    print(non_raining_false)

    raining_gt = [False for value in output1] + [True for value in output2] + [False for value in output3] + [True for value in output4]
    raining_inf = [value for value in output1] + [value for value in output2] + [value for value in output3] + [value for value in output4] 
    f, ax = plt.subplots(1,1)
    plt.rcParams["figure.figsize"] = (3,3)
    ConfusionMatrixDisplay.from_predictions(raining_gt, raining_inf, ax=ax)
    ax.axes.set_yticklabels(["False", "True"], rotation = 90, va="center")
    plt.show()
# %% Secondary testing
if __name__ == "__main__":
    sample1 = read_inference_log("Audio results\\LOGS\\LOGS\\LOG001.TXT")
    sample2 = read_inference_log("Audio results\\LOGS\\LOGS\\LOG005.TXT")
    sample3 = read_inference_log("Audio results\\LOGS\\LOGS\\LOG006.TXT")

    output1 = [True if value[0] == "Raining" else False for value in sample1]
    output2 = [True if value[0] == "Raining" else False for value in sample2]
    output3 = [True if value[0] == "Raining" else False for value in sample3]

    # also store certainty values

    value1 = [(value[1], value[2]) for value in sample1]
    value2 = [(value[1], value[2]) for value in sample2]
    value3 = [(value[1], value[2]) for value in sample3]

    # clippings and trimmings (ignore start and end where opening and closing
    # the container result in loud noises simliar to rain shield  )
    
    output1 = output1[4:624] # No rain
    output4 = output2[:165] # rain
    output2 = output2[-14:] + output3[:19] # light rain
    # Output3 started when it was still raining (lightly) and then stopped  
    output3 = output3[20:262] # No rain

    # same for certinaty
    value1 = value1[4:624]
    value4 = value2[:165]
    value2 = value2[-14:] + value3[:19]
    value3 = value3[20:262]

    certainty1 = [abs(value[0]-value[1]) / max(abs(value[0]), abs(value[1])) for value in value1]
    certainty2 = [abs(value[0]-value[1]) / max(abs(value[0]), abs(value[1])) for value in value2]
    certainty3 = [abs(value[0]-value[1]) / max(abs(value[0]), abs(value[1])) for value in value3]
    certainty4 = [abs(value[0]-value[1]) / max(abs(value[0]), abs(value[1])) for value in value4]

    isNeg1 = [(max(value) - min(value)) / max(value) if max(value) > 0 else ((max(value) - min(value)) / abs(min((value)))) for value in value1]
    isNeg2 = [(max(value) - min(value)) / max(value) if max(value) > 0 else ((max(value) - min(value)) / abs(min((value)))) for value in value2]
    isNeg3 = [(max(value) - min(value)) / max(value) if max(value) > 0 else ((max(value) - min(value)) / abs(min((value)))) for value in value3]
    isNeg4 = [(max(value) - min(value)) / max(value) if max(value) > 0 else ((max(value) - min(value)) / abs(min((value)))) for value in value4]

    isNeg1 = [(2 if value > 2 else value) for value in isNeg1]
    isNeg2 = [(2 if value > 2 else value) for value in isNeg2]
    isNeg3 = [(2 if value > 2 else value) for value in isNeg3]
    isNeg4 = [(2 if value > 2 else value) for value in isNeg4]

    positive1 = [True if max(value) > 0 else False for value in value1]
    positive2 = [True if max(value) > 0 else False for value in value2]
    positive3 = [True if max(value) > 0 else False for value in value3]
    positive4 = [True if max(value) > 0 else False for value in value4]


    graph1 = isNeg1
    graph2 = isNeg2
    graph3 = isNeg3
    graph4 = isNeg4

    
    plt.rcParams["figure.figsize"] = (5,3)

    print("No rain sample. correct:" + str((len(output1) - sum(output1))/ len(output1)*100))
    fig, ax1 = plt.subplots()
    ax1.set_title("No Rain")
    ax1.step(range(len(output1)), output1, where='post', color='C0')
    ax1.axes.set_yticks([0,1])
    ax1.axes.set_yticklabels(["None", "Rain"], rotation = 90, va="center")
    ax1.axes.set_xlabel("Sample")
    ax1.axes.set_ylabel("Inferred Rain Level")
    # ax2 = ax1.twinx()
    # ax2.fill_between(range(len(graph1)), graph1, color='C1', alpha=0.3)
    plt.show()
    print("Light Rain sample. correct:" + str(sum(output2)/ len(output2) * 100))
    fig, ax1 = plt.subplots()
    ax1.set_title("Light Rain")
    ax1.step(range(len(output2)), output2, where='post', color='C0')
    ax1.axes.set_yticks([0,1])
    ax1.axes.set_yticklabels(["None", "Rain"], rotation = 90, va="center")
    ax1.axes.set_xlabel("Sample")
    ax1.axes.set_ylabel("Inferred Rain Level")
    # ax2 = ax1.twinx()
    # ax2.fill_between(range(len(graph2)), graph2, color='C1', alpha=0.3)
    plt.show()
    print("No rain sample. correct:" + str((len(output3) - sum(output3))/ len(output3)*100))
    fig, ax1 = plt.subplots()
    ax1.set_title("No Rain")
    ax1.step(range(len(output3)), output3, where='post', color='C0')
    ax1.axes.set_yticks([0,1])
    ax1.axes.set_yticklabels(["None", "Rain"], rotation = 90, va="center")
    ax1.axes.set_xlabel("Sample")
    ax1.axes.set_ylabel("Inferred Rain Level")
    # ax2 = ax1.twinx()
    # ax2.fill_between(range(len(graph3)), graph3, color='C1', alpha=0.3)
    plt.show()
    print("Rain sample. correct:" + str(sum(output4)/ len(output4) * 100))
    fig, ax1 = plt.subplots()
    ax1.set_title("Rain")
    ax1.step(range(len(output4)), output4, where='post', color='C0')
    ax1.axes.set_yticks([0,1])
    ax1.axes.set_yticklabels(["None", "Rain"], rotation = 90, va="center")
    ax1.axes.set_xlabel("Sample")
    ax1.axes.set_ylabel("Inferred Rain Level")
    # ax2 = ax1.twinx()
    # ax2.fill_between(range(len(graph4)), graph4, color='C1', alpha=0.3)
    plt.show()
    
    print("==============================================")
    window = 15
    timeVaried1 = [sum(output1[i:i+window])/window for i in range(len(output1)-window)]
    timeVaried2 = [sum(output2[i:i+window])/window for i in range(len(output2)-window)]
    timeVaried3 = [sum(output3[i:i+window])/window for i in range(len(output3)-window)]
    timeVaried4 = [sum(output4[i:i+window])/window for i in range(len(output4)-window)]

    # for i in range(window):
    #     timeVaried1.insert(0,0)
    #     timeVaried2.insert(0,0)
    #     timeVaried3.insert(0,0)
    #     timeVaried4.insert(0,0)

    graph1 = timeVaried1
    graph2 = timeVaried2
    graph3 = timeVaried3
    graph4 = timeVaried4

    outTrimmed1 = output1[15:]
    outTrimmed2 = output2[15:]
    outTrimmed3 = output3[15:]
    outTrimmed4 = output4[15:]

    print("No rain sample. correct:" + str((len(output1) - sum(output1))/ len(output1)*100))
    fig, ax1 = plt.subplots()
    ax1.set_title("No Rain")
    ax1.step(range(len(outTrimmed1)), outTrimmed1, where='post', color='C0', alpha=0.7)
    ax1.fill_between(range(len(graph1)), graph1, color='C1', alpha=0.3)
    ax1.axes.set_yticks([0,1])
    ax1.axes.set_yticklabels(["None", "Rain"], rotation = 90, va="center")
    ax1.axes.set_xlabel("Sample")
    ax1.axes.set_ylabel("Inferred Rain Level")
    ax2 = ax1.twinx() 
    ax2.axes.set_ylabel("15 point moving average score (%)")
    ax2.axes.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax2.axes.set_yticklabels(["0", "20", "40", "60", "80", "100"])
    plt.show()
    print("Light Rain sample. correct:" + str(sum(output2)/ len(output2) * 100))
    fig, ax1 = plt.subplots()
    ax1.set_title("Light Rain")
    ax1.step(range(len(outTrimmed2)), outTrimmed2, where='post', color='C0', alpha=0.7)
    ax1.fill_between(range(len(graph2)), graph2, color='C1', alpha=0.3)
    ax1.axes.set_yticks([0,1])
    ax1.axes.set_yticklabels(["None", "Rain"], rotation = 90, va="center")
    ax1.axes.set_xlabel("Sample")
    ax1.axes.set_ylabel("Inferred Rain Level")
    ax2 = ax1.twinx() 
    ax2.axes.set_ylabel("15 point moving average score (%)")
    ax2.axes.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax2.axes.set_yticklabels(["0", "20", "40", "60", "80", "100"])
    plt.show()
    print("No rain sample. correct:" + str((len(output3) - sum(output3))/ len(output3)*100))
    fig, ax1 = plt.subplots()
    ax1.set_title("No Rain")
    ax1.step(range(len(outTrimmed3)), outTrimmed3, where='post', color='C0', alpha=0.7)
    ax1.fill_between(range(len(graph3)), graph3, color='C1', alpha=0.3)
    ax1.axes.set_yticks([0,1])
    ax1.axes.set_yticklabels(["None", "Rain"], rotation = 90, va="center")
    ax1.axes.set_xlabel("Sample")
    ax1.axes.set_ylabel("Inferred Rain Level")
    ax2 = ax1.twinx() 
    ax2.axes.set_ylabel("15 point moving average score (%)")
    ax2.axes.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax2.axes.set_yticklabels(["0", "20", "40", "60", "80", "100"])
    plt.show()
    print("Rain sample. correct:" + str(sum(output4)/ len(output4) * 100))
    fig, ax1 = plt.subplots()
    ax1.set_title("Rain")
    ax1.step(range(len(outTrimmed4)), outTrimmed4, where='post', color='C0', alpha=0.7)
    ax1.fill_between(range(len(graph4)), graph4, color='C1', alpha=0.3)
    ax1.axes.set_yticks([0,1])
    ax1.axes.set_yticklabels(["None", "Rain"], rotation = 90, va="center")
    ax1.axes.set_xlabel("Sample")
    ax1.axes.set_ylabel("Inferred Rain Level")
    ax2 = ax1.twinx() 
    ax2.axes.set_ylabel("15 point moving average score (%)")
    ax2.axes.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax2.axes.set_yticklabels(["0", "20", "40", "60", "80", "100"])
    plt.show()

    # Output 4 has lots of noise (since it will also include switching file)
    # and will not be used for monitoring.
    raining_true = sum(output4) # True Positive
    raining_false = sum(output1) + sum(output3) # False Positive
    non_raining_true = len(output1) - sum(output1) + len(output3) - sum(output3) # True Negative
    non_raining_false = len(output4) - sum(output4) # False Negative
    print("True Positives:", raining_true)
    print("False Positives: ", raining_false)
    print("True Negatives:", non_raining_true)
    print("False Positives:", non_raining_false)

    raining_gt = [False for value in output1] + [False for value in output3] + [True for value in output4]
    raining_inf = [value for value in output1] + [value for value in output3] + [value for value in output4] 
    f, ax = plt.subplots(1,1)
    plt.rcParams["figure.figsize"] = (4,4)
    ConfusionMatrixDisplay.from_predictions(raining_gt, raining_inf, ax=ax)
    ax.axes.set_yticklabels(["False", "True"], rotation = 90, va="center")
    plt.show()

# %%
# perform batch testing for selected sets
