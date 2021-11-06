

def trimSilence(samples: list, threshold=1000, duration=50) -> list:
    result = []
    start = None
    end = None
    count = 0
    started = False
    for i in range(len(samples)):
        value = samples[i]
        # skip to the start
        if not started and abs(value) < threshold:
            continue
        if abs(value) >= threshold:
            # start or continuation of a new sound
            if start is None:
                start = i
            if count != 0:
                count = 0 
            end = None

        if abs(value) < threshold:
            # ignore if there's no start
            if start is None:
                continue

            count = count + 1
            # set end if unset
            if end is None:
                end = i - 1
            
            # check silence count threshold
            if count >= threshold:
                # silence - store old and prepare for next sound
                result.append(samples[start:end])
                start = None
    return result      

def sample(samples: list, duration=2000) -> list:
    # split up the list into sub lists, first split by trimming
    result = trimSilence(samples)
    if len(result) < 1:
        result = trimSilence(samples, threshold=500)
        if len(result < 1):
            return None
    
    # now sample remaining clips.
    longClips = []
    shortClips = []
    for clip in result:
        if len(clip) >= duration:
            longClips.append(clip)
        else:
            shortClips.append(clip)

    # trim long clips into short clip
    start = 0
    end = 0
    for clip in longClips:
        if len(clip) < duration  * 2:
            # only 1 clip will fit (unless overlap occurs)
            # To-Do get a random value from 0 to (length - duration)
            offset = 0
            subclip = clip[offset : duration + offset]
            shortClips.append(subclip)

        else:
            # To-Do muliti-sample section
            subclip = clip[offset : duration + offset]
            shortClips.append(subclip)

    return shortClips