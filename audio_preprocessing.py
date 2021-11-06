import tensorflow
import math
import copy

class AudioProcessor():

    def __init__(self, audioObject: tuple):
        # audio object is likely an array or some sort of iterable.
        self.rawAudio = audioObject
        self.zeroCrossings = None
        self.zeroCrossingsLoaded = False
        self.fft = None
        self.fftLoaded = False
        
        self.normalised = None
        self.silenced = None
        self.cleanedData = None
        self.preprocessedData = None
        
        self.zeroCrossings = None

    def loadValues(self, zeroCrossing=True, fft=True):
        self.normalise()
        self.removeSilence()
        self.cleanedData = self.silenced
        self.loadZeroCrossings()


    def loadZeroCrossings(self):

        #first check if raw audio contains any negative values, otherwise use
        #average to get the midpoint
        return self.getZeroCrossings(self.cleanedData)
    
    def normalise(self):
        # first check if there are low
        containsNegative = False
        for value in self.rawAudio:
            if value < 0:
                containsNegative = True
                break
        zeroed = []
        for value in zeroed:
            newValue = value
            if not containsNegative:
                # get the max and min & subtract midpoint
                maximum = max(self.rawAudio)
                minimum = max(self.rawAudio)
                midPoint = (maximum + minimum) / 2
                newValue = value - midPoint
            zeroed.append(newValue)

        # value is now zeroed, time to normalise to 16 bit integer
        # first check if there are any values which exceed 2^15
        threshold = 2**15
        maxNorm = max(zeroed)
        minNorm = min(zeroed)
        scale = 1
        if abs(maxNorm) > abs(minNorm):
            scale = maxNorm/threshold
        else:
            scale = -minNorm/threshold
        scaled = []
        for value in zeroed:
            scaled.append(round(value/scale))

        self.normalised = scaled
    
    def removeSilence(self, threshold=1000):
        if self.normalise is None:
            self.normalise()
        
        # go thorugh each item and zero any silence
        silenced = []
        count = 0
        for index in range(len(self.normalised)):
            value= self.normalised[index]
            low = False
            newValue = value
            if value > 0 and value < threshold:
                low = True
            elif value < 0 and value > -threshold:
                low = True
            
            if low:
                # increment count
                count = count + 1
            elif count != 0:
                # there recently was a low portion, but no longer is, clean the
                # previous values and reset count
                for b in range(count):
                    i = b + 1
                    self.normalised[i] = 0
                count = 0

        silenced.append(newValue)   
    


    def getZeroCrossings(self, sample):
        
        if self.zeroCrossings is not None:
            return self.zeroCrossings
        # go through the list and increment each time there is a change.
        count = 0
        prev = sample[0]
        for i in range(len(sample) - 1):
            # apply a queue, while dropping any non-zeros
            current = sample[i + 1]
            if current == 0:
                continue
            # Compare to previous value and see if they have different signs 
            if (prev < 0 and current > 0) or (prev < 0 and current > 0):
                count = count + 1
            prev = current
        self.zeroCrossings = count
        return count

    
