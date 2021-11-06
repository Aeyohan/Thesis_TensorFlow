
#%%
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

import numpy as np
from PIL import Image

import os
import pickle

from tensorflow.python.ops.gen_math_ops import real

def round_sobel():
    # read image
    imagePath = "./Images/flickr_wiper crop.jpg"
    image_bytes = tf.io.read_file(imagePath)

    # create image object in TF
    image = tf.image.decode_image(image_bytes)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, 0)

    # Create Sobel object & perform the y Sobel
    sobel = tf.image.sobel_edges(image)
    print(sobel)
    sobel_x = np.asarray(sobel[0, :, :, :, 1]) # sobel in x-direction
    sobel_y = np.asarray(sobel[0, :, :, :, 0]) # sobel in y-direction
    cumulative = (abs(sobel_x) + abs(sobel_y)) / 2
    print(cumulative)
    plt.imshow(cumulative, cmap='hot', interpolation='nearest')
    plt.show()
    # remove any below threshold entries
    scale = cumulative.max() - cumulative.min()
    offset = -cumulative.min()
    normalised = [[(j[0] + offset) / scale for j in element] for element in cumulative]
    plt.imshow(normalised, cmap='hot', interpolation='nearest')
    plt.show()
    test_min = [[1 if j > 0.2 else 0 for j in element] for element in normalised]
    perc = 0
    total = 0
    for index in test_min:
        for value in index:
            if value != 0:
                perc = perc + 1
            total = total + 1
    print(str(perc / total * 100) + "%" + " edge @ " + str(scale))
    plt.imshow(test_min, cmap='hot', interpolation='nearest')
    plt.show()


    # display
    # Image.fromarray(sobel_y[..., 0] / 4 + 0.5).show()
    # Image.fromarray(sobel_x[..., 0] / 4 + 0.5).show()
    # Image.fromarray(cumulative[..., 0] / 4 + 0.5).show()
    
    pass

def load_image_set(folderPath, setFilter, keep_raws=False):
    files = get_file_list(folderPath, setFilter)
    raws = []
    # load images using tfio
    images = [tf.io.read_file(file) for file in files]
    if keep_raws:
        raws = [tf.image.decode_image(image) for image in images]
    images = [tf.image.rgb_to_grayscale(tf.image.decode_image(image)) for image in images]
    images = [tf.cast(image, tf.float32) for image in images]
    images = [tf.expand_dims(image, 0) for image in images]
    # Images should be 640x320, crop to 320x320
    edge = round((569 - 320) / 2) # should be 105 
    images = [tf.image.crop_to_bounding_box(image, 0, edge, 320, 320) for image in images]
    if keep_raws:
        raws = [tf.image.crop_to_bounding_box(image, 0, edge, 320, 320) for image in raws]
        return (images,raws)
    return images

def apply_sobel(imageSet):
    sobel = [abs(tf.image.sobel_edges(image)) for image in imageSet]
    edge_sum = [tf.reduce_mean(image, 4) for image in sobel]
    reshape = [tf.reshape(image, image.shape[0:3]) for image in edge_sum]

    return reshape

def apply_partition(imageSet, partitions=[42, 1907]):
    result = []
    result.append(imageSet[:partitions[0]])
    for i in range(len(partitions)-1):
        result.append(imageSet[partitions[i]:partitions[i+1]])
    result.append(imageSet[partitions[-1]:])
    return result

def save_cache_set(imageSet, folderPath, setFilter):
    filePath = "image_set_" + setFilter + ".pic"
    f = open(filePath, 'wb')
    pickle.dump((folderPath, imageSet), f)
    f.close()

def load_cache_set(folderPath, setFilter):
    filePath = "image_set_" + setFilter + ".pic"
    if os.path.exists(filePath):
        f = open(filePath, 'rb')
        result = pickle.load(f)
        f.close()
        if result[0] == folderPath:
            return result[1]
    return None

def load_sobel(folderPath="E:\Data\Thesis\images\still_frames\scaled_down", setFilter="held"):
    imageSet = load_cache_set(folderPath, setFilter)
    if imageSet is None:
        imageSet = load_image_set(folderPath, setFilter)
        sobelSet = apply_sobel(imageSet)
        # sobelSets = apply_partition(sobelSet)
        save_cache_set(sobelSet, folderPath, setFilter)
        return sobelSet
    return imageSet

def sample(imageSet, short_duration=1, short_count=2, long_duration=59):
    # sample short_count images every short seconds and then sleep for long seconds
    result = []
    i = 0
    while i + (short_count * short_duration) < len(imageSet):
        snapshot = []
        for count in range(short_count):
            snapshot.append(imageSet[i + count * short_duration])
        i += short_count * short_duration
        i += long_duration
        result.append(tf.concat(snapshot, 0))
    return result
    

def get_file_list(rootFolderPath, setFilter):
    # create an array of tupples, the tupple contains the path with file
    # contents as another element
    filePaths = []
    for root, folders, files in os.walk(rootFolderPath):
        for fileName in files:
            if fileName.__contains__(setFilter) and (fileName.__contains__(".png") or fileName.__contains__(".jpg") or fileName.__contains__(".jpeg")):
                filePaths.append(os.path.join(root, fileName))
    print("found " + str(len(filePaths)) + " png files")
    return filePaths

#%%
if __name__ == "__main__":
    imageSet = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "normal")

    images = load_sobel(setFilter="normal")
    samples = sample(images)
    images = [tf.reshape(image, [320,320]) for image in images]
    print("sobels")
    plt.imshow(images[0].numpy(), cmap='hot', interpolation='nearest')
    plt.show()
    plt.imshow(images[1].numpy(), cmap='hot', interpolation='nearest')
    plt.show()
    plt.imshow(images[2].numpy(), cmap='hot', interpolation='nearest')
    plt.show()
    plt.imshow(images[3].numpy(), cmap='hot', interpolation='nearest')
    plt.show()
    
    

    # i1 = images[0]
    # i2 = images[1]
    # i3 = images[2]
    # i4 = images[3]

    
    # thresh = 10
    # b1 = [[1 if j > thresh else 0 for j in element] for element in i1]
    # b2 = [[1 if j > thresh else 0 for j in element] for element in i2]
    # b3 = [[1 if j > thresh else 0 for j in element] for element in i3]
    # print("bitwise")
    # plt.imshow(b1, cmap='hot', interpolation='nearest')
    # plt.show()

    # print("xor")
    # c1 = [[j[0] ^ j[1] for j in zip(element[0], element[1])] for element in zip(b1, b2)]
    # c2 = [[j[0] ^ j[1] for j in zip(element[0], element[1])] for element in zip(b2, b3)]
    # plt.imshow(c1, cmap='hot', interpolation='nearest')
    # plt.show()
    # plt.imshow(c2, cmap='hot', interpolation='nearest')
    # plt.show()

# %%
def get_frame_changes(samples):
    percentages = []
    images =[]
    # get mean of each frame
    samples = [tf.reshape(tf.reduce_mean(sample, 0), [320,320]) for sample in samples]
    
    # get the difference
    differences = []
    for i in range (len(samples) - 1):
        differences.append(abs(samples[i] - samples[i + 1]))
    thresh = 10
    bit_differences = [tf.where(frame > thresh, 1, 0) for frame in differences]

    for bitDiff in bit_differences:
        # iterate through the image and grab how many 1's there are & represent
        # as a percentage
        count = 0
        image = bitDiff.numpy()
        for row in image:
            for val in row:  
                if val == 1:
                    count += 1

        percentages.append(count / 320**2 * 100)
        images.append(bitDiff)
    return (percentages, images)

# %%
if __name__ == "__main__":
    f1 = tf.reshape(tf.reduce_mean(samples[0], 0),[320,320])
    f2 = tf.reshape(tf.reduce_mean(samples[1], 0),[320,320])
    print("frame 1 & 2")
    plt.imshow(f1, cmap='hot', interpolation='nearest')
    plt.show()
    plt.imshow(f2, cmap='hot', interpolation='nearest')
    plt.show()

    print("difference")
    d1 = abs(f1-f2)
    plt.imshow(d1, cmap='hot', interpolation='nearest')
    plt.show()

    print("bitwise")
    thresh = 10
    # b1 = [[1 if j > thresh else 0 for j in element] for element in d1.numpy()]
    b1 = tf.where(d1 > thresh, 1, 0)
    plt.imshow(b1, cmap='hot', interpolation='nearest')
    plt.show()

    results = get_frame_changes(samples)
    

# %%
if __name__ == "__main__":
    coverage, images = get_frame_changes(samples)
    
    print(coverage)
    for image in images:
        plt.imshow(image, cmap='hot', interpolation='nearest')
        plt.show()

# %%

def get_sequence_deltas(samples):
    # first remove any short time variances by performing an and operation
    percentages = []
    images =[]
    # get mean of each frame
    samples = [tf.reshape(tf.reduce_mean(sample, 0), [320,320]) for sample in samples]

def plot(data):
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.show()

def get_improved_deltas(samples, thresh=10):
    percentages = []
    images =[]
    # first quantise frames to booleans (1 and 0)
    quantised = [tf.where(sample > thresh, 1, 0) for sample in samples]

    # second remove any short time variances by performing an & operation   
    frames = [frame[0] & frame[1] for frame in quantised]

    # now to perform inter-frame comparisons using xor (^) to find the
    # differences
    
    for i in range(len(frames) - 1):
        diff = frames[i] ^ frames[i + 1]
        percentages.append((tf.reduce_sum(diff) / (320 * 320)))
        images.append(diff)

    return (percentages, images)
# %%
images = load_sobel(setFilter="macro_0")
images += load_sobel(setFilter="macro_1")
samples = sample(images)
coverage, images = get_improved_deltas(samples)

output = [True if sample > 0.17 else False for sample in coverage]

# %% Display_1
if __name__ == "__main__":
    try:
        rawImageSet
    except:
        rawImageSet = sample(load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "normal"))
        
    
    sobelSet = load_sobel(setFilter="normal")

    rawImage_sampled = [tf.split(rawImage, num_or_size_splits=2, axis=0)[0] for rawImage in rawImageSet]
    seq = 0
    for image, sobel in zip(rawImage_sampled, sobelSet):
        f, ax = plt.subplots(1, 3)
        # add data
        ax[0].imshow(tf.reshape(image, [320,320]), cmap='gray', interpolation='nearest')
        ax[0].set_title('Grayscale Image')
        ax[1].imshow(tf.reshape(sobel, [320,320]), cmap='gray', interpolation='nearest')
        ax[1].set_title('Grayscale Sobel')
        ax[2].imshow(tf.reshape(sobel, [320,320]), cmap='hot', interpolation='nearest')
        ax[2].set_title('High Contrast Sobel')
        # turn off axes
        for axis in ax:
            axis.axis('off')
            # resize for a better scaled image
        plt.rcParams["figure.figsize"] = (15,5)
        plt.savefig("imageProcessingGraphs\\seq1_" + str(seq) +".png", bbox_inches='tight')
        plt.show()
        
        seq += 1

# %%
def get_frame_changes_debug(samples):
    percentages = []
    images =[]
    # get mean of each frame
    thresh = 10
    bit_short_time = [tf.where(frame > thresh, 1, 0) for frame in samples]

    bit_im = [(tf.reshape(tf.split(sample, num_or_size_splits=2, axis=0)[0], shape=[320,320]), tf.reshape(tf.split(sample, num_or_size_splits=2, axis=0)[0], shape=[320,320])) for sample in bit_short_time]
    
    bit_short_time = [tf.reshape(tf.split(sample, num_or_size_splits=2, axis=0)[0], shape=[320,320]) & tf.reshape(tf.split(sample, num_or_size_splits=2, axis=0)[1], shape=[320,320]) for sample in bit_short_time]

    # perform long time 
    differences = []
    for i in range(len(bit_short_time) - 1):
        differences.append(bit_short_time[i] ^ bit_short_time[i + 1])

    for element in differences:
        percentages.append(element.numpy().sum() / (320**2) * 100)
        images.append(element)

    return (percentages, images, bit_short_time, bit_im)
    



    # samples = [tf.reshape(tf.reduce_mean(sample, 0), [320,320]) for sample in samples]
    
    # # get the difference
    # differences = []
    # for i in range (len(samples) - 1):
    #     differences.append(abs(samples[i] - samples[i + 1]))
    
    # bit_differences = [tf.where(frame > thresh, 1, 0) for frame in differences]

    # for bitDiff in bit_differences:
    #     # iterate through the image and grab how many 1's there are & represent
    #     # as a percentage
    #     count = 0
    #     image = bitDiff.numpy()
    #     for row in image:
    #         for val in row:  
    #             if val == 1:
    #                 count += 1

    #     percentages.append(count / 320**2 * 100)
    #     images.append(bitDiff)
    # return (percentages, images, bit_differences, differences)
# %% Display 2
if __name__ == "__main__":
    # try:
    #     rawImageSet
    # except:
    #     rawImageSet = sample(load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "normal"))
    images = load_sobel(setFilter="normal")
    samples = sample(images)

    perc, im, bit_s, bit_q = get_frame_changes_debug(samples)

    split_pair_samples = [tf.split(sample, num_or_size_splits=2, axis=0)[0] for sample in samples]

    seq = 0
    for im, s, q, source in zip(im, bit_s[1:], bit_q[1:], split_pair_samples):
        f, ax = plt.subplots(2, 2)
        # add data
        ax[0][0].imshow(tf.reshape(source, [320,320]), cmap='gray', interpolation='nearest')
        ax[0][0].set_title('Grayscale Sobel')
        ax[0][1].imshow(tf.reshape(q, [320,320]), cmap='gray', interpolation='nearest')
        ax[0][1].set_title('Bit Quantised Sobel')
        ax[1][0].imshow(tf.reshape(s, [320,320]), cmap='gray', interpolation='nearest')
        ax[1][0].set_title('Short Time Static')
        ax[1][1].imshow(tf.reshape(im, [320,320]), cmap='hot', interpolation='nearest')
        ax[1][1].set_title('Long Time differences')
        # turn off axes
        for axis in ax:
            for subax in axis:
                subax.axis('off')
            # resize for a better scaled image
        plt.rcParams["figure.figsize"] = (6,6)
        plt.savefig("imageProcessingGraphs\\seq2_" + str(seq) +".png", bbox_inches='tight')
        plt.show()
        
        seq += 1
# %%

# Results 
if __name__ == "__main__":
    # load macro in 2 stages or it crashes the gpu since it only has 3gb
    images = load_sobel(setFilter="macro_0") 
    images += load_sobel(setFilter="macro_1")
    samples = sample(images, long_duration=29)

    perc, im, bit_s, bit_q = get_frame_changes_debug(samples)

    print(perc)

    # validation
    times = []
    for i in range(len(perc)):
        # Samples start a t=32 and samples are every 60s
        times.append(32 + i * 30) 
    labels = {0:"No Rain", 1:"Light rain", 2:"Rain"}
    # sample = [(start, end), type]
    rainStates = [[(39,947),0], [(948,955), 1], [(956, 1160), 2], [(1161, 1470),1]] # unique for macro
    
    lightRainThresh = 15
    rainThresh = 20

    raining = []
    rainType = []

    results = []

    for time, value in zip(times[1:], perc):
        # check the time and compare to the label

        # find the current rain status
        for state in rainStates:
            if time == state[0][0] or time == state[0][1] or (time > state[0][0] and time < state[0][1]):
                # current time
                raining = state[1] != 0
                rainType = state[1]
                inferredState = 0 if value < lightRainThresh else (2 if value > rainThresh else 1)
                results.append([time, raining, rainType, inferredState])
            continue
    real_time = []
    real_rain = []

    for i in range(39, 1470):
        real_time.append(i)
        for state in rainStates:
            if i == state[0][0] or i == state[0][1] or (i > state[0][0] and i < state[0][1]):
                real_rain.append(state[1])
            continue

    rain_time = []
    rain_est = []
    for result in results:

        print("At time " + str(result[0]) + ", there was: " + labels[result[2]] + ", the model deduced: " + labels[result[3]])
        rain_time.append(result[0])
        rain_est.append(result[3])
    plt.rcParams["figure.figsize"] = (9,3)
    f, ax = plt.subplots(1,1)
    
    ax.fill_between(real_time, real_rain, color='C0', alpha=0.3)
    ax.plot(rain_time, rain_est ,color='C1', alpha=0.7)
    ax.set_title("Image recognition Sample")
    ax.axes.set_xlabel("Time (s)")
    ax.axes.set_ylabel("Rain Level")
    ax.axes.set_yticks([0,1,2])
    ax.axes.set_yticklabels(["None", "Light Rain", "Rain"], rotation = 90, va="center")
    ax.legend(["Device Inferred", "Actual Rain"], loc='upper center')
    plt.show()
# %%



if __name__ == "__main__":
    # load macro in 2 stages or it crashes the gpu since it only has 3gb
    images = load_sobel(setFilter="sunny1") 
    # images += load_sobel(setFilter="macro_1")
    samples = sample(images, long_duration=29)

    perc, im, bit_s, bit_q = get_frame_changes_debug(samples)

    print(perc)
    lightRainThresh = 15
    inferredState = [value >= lightRainThresh for value in perc]
    
    print("Sunny1 has " + str(len(perc)) + " readings with " + str(len(perc) - sum(inferredState)) + " incorrect readings")
    images = load_sobel(setFilter="sunny2") 
    # images += load_sobel(setFilter="macro_1")
    samples = sample(images, long_duration=29)

    perc, im, bit_s, bit_q = get_frame_changes_debug(samples)
    inferredState = [value >= lightRainThresh for value in perc]

    print("Sunny2 has " + str(len(perc)) + " readings with " + str(len(perc) - sum(inferredState)) + " incorrect readings")

# %%

# raw results
if __name__ == "__main__":
    # load macro in 2 stages or it crashes the gpu since it only has 3gb
    low_light = True
    long_duration = 14
    if low_light:
        images = load_sobel(setFilter="low_light_0")
        images += load_sobel(setFilter="low_light_1")
    else:
        images = load_sobel(setFilter="macro_0") 
        images += load_sobel(setFilter="macro_1")
    samples = sample(images, long_duration=long_duration)

    perc, im, bit_s, bit_q = get_frame_changes_debug(samples)
    
    if low_light:
        try:
            low_light_images
        except:
            low_light_images = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "low_light_0")
            low_light_images += load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "low_light_1")
        ll_samples = sample(low_light_images, long_duration=long_duration)
        thresh = 10
        # bit_short_time = [tf.where(frame[0] > thresh, 1, 0) for frame in ll_samples]
        light_level = [frame.numpy().sum() / (320**2 * 256) * 100 for frame in low_light_images]

    print(perc)

    # validation
    times = []
    for i in range(len(perc)):
        # Samples start a t=32 and samples are every 60s
        if low_light:
            times.append(33 + i * (long_duration + 1))
        else:
            times.append(39 + i * (long_duration + 1)) 
    
    # sample = [(start, end), type]
    if low_light:
        rainStates = [[(0,375), 0], [(376, 1400), 1]]
        labels = {0:"No Rain", 1:"Rain"}
    else:
        rainStates = [[(39,947),0], [(948,955), 1], [(956, 1160), 2], [(1161, 1470),1], [(1471, 1896),0]] # unique for macro
        labels = {0:"No Rain", 1:"Light Rain", 2:"Rain"}

    lightRainThresh = 15
    rainThresh = 20

    classified = []
    if not low_light:
        classified = [0 if value < lightRainThresh else (2 if value > rainThresh else 1) for value in perc]

    raining = []
    rainType = []

    results = []

    for time, value in zip(times, perc):
        # check the time and compare to the label

        # find the current rain status
        for state in rainStates:
            if time == state[0][0] or time == state[0][1] or (time > state[0][0] and time < state[0][1]):
                # current time
                raining = state[1] != 0
                rainType = state[1]
                inferredState = value
                results.append([time, raining, rainType, inferredState])
            continue
    real_time = []
    real_rain = []
    if low_light:
        ranges = (33, 1401)
    else:
        ranges = (39, 1896)
    for i in range(ranges[0], ranges[1]):
        real_time.append(i)
        triggered = False
        for state in rainStates:
            if i == state[0][0] or i == state[0][1] or (i > state[0][0] and i < state[0][1]):
                real_rain.append(state[1])
                triggered = True
            continue
        if not triggered:
            print("Missed State " + str(i))

    rain_time = []
    rain_est = []
    for result in results:

        # print("At time " + str(result[0]) + ", there was: " + labels[result[2]] + ", the model deduced: " + str(result[3])  + "%")
        rain_time.append(result[0])
        rain_est.append(result[3])
    plt.rcParams["figure.figsize"] = (12,4)
    f, ax = plt.subplots(1,1)
    lines = []
    lines.append(ax.fill_between(real_time, real_rain, color='C0', alpha=0.3, label="Actual Rain"))
    ax2 = ax.twinx() 
    
    if low_light:
        scale = max(light_level)
        plot_light_level = [value / scale for value in light_level]
        lines += ax.step(real_time, plot_light_level, where='post', color='C1', alpha=0.7, label="Noramlised Light Level")
        lines += ax2.step(rain_time, rain_est, color='C2', alpha=0.7, label="Rain Score")
    else:
        # lines += ax.step(times, classified, where='post', color='C1', alpha=0.7, label="Inferred Class")
        lines += ax2.step(rain_time, rain_est, where='post', color='C2', alpha=0.7, label="Rain Score")
        
    ax2.axes.set_ylabel("Rain Detection Score (%)")
    if low_light:
        ax.set_title("Image Recognition Sample B (Low Light Performance)")
    else:
        ax.set_title("Image recognition Sample A (" + str(long_duration + 1) + " delay interval)")
    ax.axes.set_xlabel("Time (s)")
    ax.axes.set_ylabel("Rain Level")
    if low_light:
        ax.axes.set_yticks([0,1])
        ax.axes.set_yticklabels(["None", "Rain"], rotation = 90, va="center")
    else:
        ax.axes.set_yticks([0,1,2])
        ax.axes.set_yticklabels(["None", "Light Rain", "Rain"], rotation = 90, va="center")
    legend_labels = [line.get_label() for line in lines]
    ax.legend(lines, legend_labels, loc='upper right')
    
    plt.show()

        
# %%
# pipeline graphics

if __name__ == "__main__":
    # load macro in 2 stages or it crashes the gpu since it only has 3gb
    low_light = True
    long_duration = 29
    sample_start = 34 # used for mapping second based data to sampled data
    if low_light:
        images = load_sobel(setFilter="low_light_0")
        images += load_sobel(setFilter="low_light_1")
    else:
        images = load_sobel(setFilter="macro_0") 
        images += load_sobel(setFilter="macro_1")
    samples = sample(images, long_duration=long_duration)

    perc, im, bit_s, bit_q = get_frame_changes_debug(samples)
    
    raws = []
    grayscale = []

    if low_light:
        try:
            low_light_images
            grayscale = low_light_images
            raws = low_light_raws
        except:
            low_light_images, low_light_raws = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "low_light_0", keep_raws=True)
            temp = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "low_light_1", keep_raws=True)
            low_light_images += temp[0]
            low_light_raws += temp[1]
            grayscale = low_light_images
            raws = low_light_raws
    else:
        try:
            normal_images
            grayscale = normal_images
            raws = normal_raws
        except:
            (normal_images, normal_raws) = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "macro_00", keep_raws=True)
            temp = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "macro_01", keep_raws=True)
            normal_images += temp[0]
            normal_raws += temp[1]
            temp = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "macro_02", keep_raws=True)
            normal_images += temp[0]
            normal_raws += temp[1]
            temp = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "macro_03", keep_raws=True)
            normal_images += temp[0]
            normal_raws += temp[1]
            temp = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "macro_04", keep_raws=True)
            normal_images += temp[0]
            normal_raws += temp[1]
            temp = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "macro_05", keep_raws=True)
            normal_images += temp[0]
            normal_raws += temp[1]
            temp = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "macro_06", keep_raws=True)
            normal_images += temp[0]
            normal_raws += temp[1]
            temp = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "macro_07", keep_raws=True)
            normal_images += temp[0]
            normal_raws += temp[1]
            temp = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "macro_08", keep_raws=True)
            normal_images += temp[0]
            normal_raws += temp[1]
            temp = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "macro_09", keep_raws=True)
            normal_images += temp[0]
            normal_raws += temp[1]
            
            temp = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "macro_10", keep_raws=True)
            normal_images += temp[0]
            normal_raws += temp[1]
            temp = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "macro_11", keep_raws=True)
            normal_images += temp[0]
            normal_raws += temp[1]
            temp = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "macro_12", keep_raws=True)
            normal_images += temp[0]
            normal_raws += temp[1]
            temp = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "macro_13", keep_raws=True)
            normal_images += temp[0]
            normal_raws += temp[1]
            temp = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "macro_14", keep_raws=True)
            normal_images += temp[0]
            normal_raws += temp[1]
            temp = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "macro_15", keep_raws=True)
            normal_images += temp[0]
            normal_raws += temp[1]
            temp = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "macro_16", keep_raws=True)
            normal_images += temp[0]
            normal_raws += temp[1]
            temp = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "macro_17", keep_raws=True)
            normal_images += temp[0]
            normal_raws += temp[1]
            temp = load_image_set("E:\Data\Thesis\images\still_frames\scaled_down", "macro_18", keep_raws=True)
            normal_images += temp[0]
            normal_raws += temp[1]
            grayscale = normal_images
            raws = normal_raws

    
    step = long_duration
    unsampled_indexes = [sample_start * step, sample_start * step + 1, sample_start * (step + 1), sample_start * (step + 1)]
    
    sampled_indexes = [sample_start, sample_start + 1]


    # for the pipeline, we require two Standard images, the post Greyscale,
    # Sobel, Quantised, Short time and long time combination.

    raw_images = [raws[i] for i in unsampled_indexes] # raw cropped RGB
    grayscale_images = [tf.reshape(grayscale[i], [320,320]) for i in unsampled_indexes] # grayscale 

    sobel_images = [samples[i] for i in sampled_indexes] # Sobel
    quantised_images = [bit_q[i] for i in sampled_indexes]
    short_images = [bit_s[i] for i in sampled_indexes]
    # Quantised

    print("RGB")
    count = 0
    for image in raw_images:
        f, ax = plt.subplots(1,1)
        ax.imshow(image)
        ax.axis('off')
        plt.savefig("imageProcessingGraphs\\images\\RGB_images_" + str(sample_start) + "_" + str(count) +".png", bbox_inches='tight')
        count += 1
        plt.show()

    print("Grayscale")
    count = 0
    for image in grayscale_images:
        f, ax = plt.subplots(1,1)
        ax.imshow(tf.reshape(image, [320,320]), cmap="gray")
        ax.axis("off")
        plt.savefig("imageProcessingGraphs\\images\\Gray_images_" + str(sample_start) + "_" + str(count) +".png", bbox_inches='tight')
        count += 1

    print("Sobel")
    count = 0
    for imageGroup in sobel_images:
        for image in imageGroup:
            f, ax = plt.subplots(1,1)
            ax.imshow(image, cmap="gray")
            ax.axis("off")
            plt.savefig("imageProcessingGraphs\\images\\Sobel_images_" + str(sample_start) + "_" + str(count) +".png", bbox_inches='tight')
            count += 1
            plt.show()

    print("Quantised")
    count = 0
    for imageGroup in quantised_images:
        for image in imageGroup:
            f, ax = plt.subplots(1,1)
            ax.imshow(image, cmap="gray")
            ax.axis("off")
            plt.savefig("imageProcessingGraphs\\images\\Quantised_images_" + str(sample_start) + "_" + str(count) +".png", bbox_inches='tight')
            count += 1
            plt.show()
            
    print("Short TIme")
    count = 0
    for image in short_images:
        f, ax = plt.subplots(1,1)
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        plt.savefig("imageProcessingGraphs\\images\\short_images_" + str(sample_start) + "_" + str(count) +".png", bbox_inches='tight')
        count += 1
        plt.show()

    print("Long Time")
    count = 0
    f, ax = plt.subplots(1,1)
    ax.imshow(im[sample_start], cmap="gray")
    ax.axis("off")
    plt.savefig("imageProcessingGraphs\\images\\Long_image" + str(sample_start) +".png", bbox_inches='tight')
    plt.show()

# %%
# Bird example (Continued) (run above ) Only use with Macro image set

if __name__ == "__main__":
    # sample_num = 215 - 32
    sample_num = 183 - 33
    sample_num = 666 - 33
    sample_num = 940 - 33
    sample_num = 1073 - 33
    # for i in range(215,260):
    print("Raw")
    plt.imshow(raws[sample_num])
    plt.axis('off')
    plt.show()
    
    print("Grayscale")
    plt.imshow(tf.reshape(grayscale[sample_num], [320,320]), cmap='gray')
    plt.axis('off')
    plt.show()

    print("Quantised")
    plt.imshow(tf.reshape(images[sample_num], [320,320]), cmap='gray')
    plt.axis('off')
    plt.show()

    print("Quantised + 1")
    plt.imshow(tf.reshape(images[sample_num + 1], [320,320]), cmap='gray')
    plt.axis('off')
    plt.show()

    print("Short Time")
    thresh = 10
    plt.imshow(tf.reshape(tf.where(images[sample_num] > thresh, 1, 0) & tf.where(images[sample_num + 1] > thresh, 1, 0), [320,320]), cmap='gray')
    plt.axis('off')
    plt.show()
# %%

