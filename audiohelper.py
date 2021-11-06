#%%
import os
import tensorflow as tf
from tensorflow.python.framework.importer import _PopulateTFImportGraphDefOptions

#%%
def load_txt_mono(path):
    file = open(path, 'r')
    # read the first line of no-value
    file.readline()
    values = [float(line)/256 for line in file]
    file.close()
    values = [1 if value > 0.99 else (-1 if value < -0.99 else value) for value in values]
    return tf.convert_to_tensor(values)

def txt_to_wav(paths):
    fileNames = [name.split("/")[-1].split("\\")[-1].split(".")[-2] for name in paths]
    values = [load_txt_mono(path) for path in paths]
    outputs = [tf.audio.encode_wav(tf.transpose(tf.stack([value,value])), 16000) for value in values]
    prefix = "E:/Data/Thesis/Audio/outputs/"
    for index in range(len(fileNames)):
        file = open(prefix + (fileNames[index]) + ".wav", 'wb')
        file.write(outputs[index].numpy())
        file.close()
# %%
if __name__ == "__main__":
    # root = "C:/Users/Aeyohan/Desktop/MouseWithoutBorders/"

    # file1 = root + "LOG007.TXT"
    # file2 = root + "LOG010.TXT"

    # fileNames = [file1, file2]

    # values = []
    # for name in fileNames:
    #     value = load_txt_mono(name)
    #     values.append(value)

    # # %%
    # output = []
    # for value in values:
    #     print(value.shape)
    #     output.append(tf.audio.encode_wav(tf.transpose(tf.stack([value,value])), 16000))
    # # %%
    # prefix = "E:/Data/Thesis/Audio/outputs/"
    # for index in range(len(fileNames)):
    #     file = open(prefix + (fileNames[index]).split("/")[-1].split("\\")[-1].split(".")[-2] + ".wav", 'wb')
    #     file.write(output[index].numpy())
    #     file.close()

    folder = "C:/Users/Aeyohan/Desktop/MouseWithoutBorders/testing data/Light rain + contaminated sound/"
    subItems = os.listdir(folder)
    for i in range(len(subItems)):
        subItems[i] = folder + subItems[i]
    txt_to_wav(subItems)

# %%

def convert_txt_to_wav(folder):

    # files should contain 000-999
    fileNames = os.listdir(folder)
    for i in range(len(fileNames)):
        fileNames[i] = folder + "/" + fileNames[i]
    # check for value entries separate values by those
    batch = 0
    itemIndex = 0
    previousIndex = 0;
    prefix = "E:/Data/Thesis/Audio/outputs/"
    rootName = folder.split("/")[-1].split("\\")[-1]    
    suffix = "_"
    destFolder = prefix + rootName + suffix + str(batch)
    try:
        os.mkdir(destFolder)
    except OSError as error:
        pass

    legacy = False
    for fileName in fileNames:
        file = open(fileName, 'r')
        header = file.readline()
        if (header.__contains__("value") and not header.__contains__("value: ")):
            # we have a legacy file
            legacy = True
        if legacy:
            index = 1
        else:
            index = int(header.split("value: ")[1])
        if fileName == "LOG999.TXT":
            # edge case, create sub-batch as this file likely will contain
            # files worth of data
            waveform = []
            for line in file:
                if line.__contains__("value"): # we now need to write the file
                    if previousIndex == 0 and index == 0:
                        # first file, normal operation
                        itemIndex = 0 # reset item index since we're in a new folder
                    elif previousIndex != 0 and index == 0:
                        # batch has reset, log current batch and create a new one
                        previousIndex = 0
                        batch += 1
                        destFolder = prefix + rootName + suffix + str(batch)
                        try:
                            os.mkdir(destFolder)
                        except OSError as error:
                            pass
                        itemIndex = 0
                         
                    # new waveform, write the old one before moving on.
                    waveForm = [1 if value > 0.99 else (-1 if value < -0.99 else value) for value in waveform]
                    # first check if the index indicates a new batch
                    fileNameFormat = f'{itemIndex:03d}'
                    dest = destFolder + "/" + str(fileNameFormat) + ".wav"
                    write_data_to_wav(tf.convert_to_tensor(waveform), dest)
                    previousIndex = index
                    itemIndex += 1
                    waveform = []
                    
                    
                    # update for the new file.
                    if legacy:
                        index += 1
                    else:
                        index = int(file.readline().split("value: ")[1])
                else:
                    value = float(line)/256
                    waveform.append(value)
        else:
            # first check if the index indicates a new batch
            if previousIndex == 0 and index == 0:
                # first file, normal operation
                itemIndex = 0 # reset item index since we're in a new folder
            elif previousIndex != 0 and index == 0:
                # batch has reset, log current batch and create a new one
                previousIndex = 0
                batch += 1
                destFolder = prefix + rootName + suffix + str(batch)
                itemIndex = 0 # reset item index since we're in a new folder
                try:
                    os.mkdir(destFolder)
                except OSError as error:
                    pass
                pass   
                
            # read as per usual
            waveform = [float(line)/256 for line in file]
            file.close()
            # clip waveform
            waveform = [1 if value > 0.99 else (-1 if value < -0.99 else value) for value in waveform]
            fileNameFormat = f'{itemIndex:03d}'
            dest = destFolder + "/" + str(fileNameFormat) + ".wav"
            write_data_to_wav(tf.convert_to_tensor(waveform), dest)
            previousIndex = index
            itemIndex += 1

    # and we're done here
    return

def write_data_to_wav(data, destinationPath):
    output = tf.audio.encode_wav(tf.transpose(tf.stack([data, data])), 16000)
    file = open(destinationPath, 'wb')
    file.write(output.numpy())
    file.close()
# %%
# Copy respective files
import pandas as pd
data = pd.read_csv("E:\\Data\\Thesis\\Audio\\outputs\\labels.csv")
srcPaths = []
destPaths = []
rootPath = "E:\\Data\\Thesis\\Audio\\outputs\\"
filteredDir = "E:\\Data\\Thesis\\Audio\\filteredOutputs\\"
for i in range(len(data["Folder"])):
    srcPaths.append(os.path.join(rootPath + data["Folder"][i], data["Sample"][i]))
    destPaths.append(os.path.join(filteredDir + data["Folder"][i], data["Sample"][i]))
import shutil
for i in range(len(srcPaths)):
    src = srcPaths[i]
    dest = destPaths[i]
    shutil.copyfile(src, dest)

# %%
