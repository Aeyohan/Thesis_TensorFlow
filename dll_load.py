# %%
from ctypes import *
import tensorflow as tf
import pickle
import numpy as np
import ctypes
# %%
frontend = cdll.LoadLibrary("./audio_frontend_wsl.so")
# %%

# frontend.FrontendProcessSamples



# # %%

# # %%
# inputDataFloat = (slices[0].numpy()).tolist()
# inputDataInt = [np.short(value) for value in inputDataFloat]
# # %%
# # %%

# # class FrontendOutput(Structure):
# #     _fields_ = [
# #         ("values", POINTER(c_uint16)),
# #         ("size", c_ulonglong)
# #     ]

# # # configure inputs
# # frontend.process_inputs.argtypes=[POINTER(c_int16), POINTER(c_ulonglong)]
# # # configure outputs
# # frontend.process_inputs.restype=FrontendOutput

# # # pre-process inputs
# # subsample = [int(inputDataInt[i]) for i in range(len(inputDataInt))]
# # values_in = (c_int16*10500) (*subsample)
# # samplesRead = [0]
# # samplesRead_in = (c_ulonglong*1) (*samplesRead)
# # res = frontend.process_inputs(values_in, samplesRead_in)



# # %%
# #configure inputs
# frontend.get_spectograph.argtypes=[POINTER(c_int16), POINTER(c_uint16), POINTER(c_ulonglong)]

# #configure output
# frontend.get_spectograph.restype=POINTER(c_uint16)
# listValues = (slices[0].numpy()).tolist()
# intValues = [int(value) for value in listValues]
# values_in = (c_int16*10500) (*intValues)
# shiftsCompleted = [0]
# shiftsCompleted_in = (c_uint16*1) (*shiftsCompleted)
# sliceSize = [0]
# sliceSize_in = (c_ulonglong*1) (*sliceSize)
# result = frontend.get_spectograph(values_in, shiftsCompleted_in, sliceSize_in)
# # %%
# print(str(shiftsCompleted_in[0]))
# print(str(sliceSize_in[0]))

# # %%
# output = [[result[i * 32 + j] for j in range(32)] for i in range(32)]
# %%

def get_spec_graph(data):
    #configure inputs types
    frontend.get_spectograph.argtypes=[POINTER(c_int16), POINTER(c_uint16), POINTER(c_ulonglong)]
    #configure output types
    frontend.get_spectograph.restype=POINTER(c_uint16)

    # configure input values
    listValues = (data.numpy()).tolist()
    intValues = [int(value) for value in listValues]
    values_in = (c_int16*10500) (*intValues)
    shiftsCompleted = [0]
    shiftsCompleted_in = (c_uint16*1) (*shiftsCompleted)
    sliceSize = [0]
    sliceSize_in = (c_ulonglong*1) (*sliceSize)
    result = frontend.get_spectograph(values_in, shiftsCompleted_in, sliceSize_in)
    return tf.convert_to_tensor(np.asarray([[result[i * 32 + j] for j in range(32)] for i in range(32)]))
# %%

def cache_spec_graphs():
    cacheFile = open("tf_slice_cache.pic", 'rb')
    data = pickle.load(cacheFile) 
    files = data[0]
    slices = data[1]
    results = []
    for i in range(len(slices)):
        element = slices[i]
        results.append(get_spec_graph(element))
        if i % 100 == 0:
            print("completed " + str(i) + " of " + str(len(slices)) + " samples")
    # once we have all the elements, save to fiel
    resultCacheFile = open("tf_spec_graph_cache.pic", 'wb')
    output = (files, results)
    data = pickle.dump(output,resultCacheFile)
    resultCacheFile.close() 
    
# %%
