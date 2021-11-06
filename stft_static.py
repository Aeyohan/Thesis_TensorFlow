# This file contains a collection of workarounds for missing TFLite support from:
# https://github.com/tensorflow/magenta/tree/master/magenta/music
# as posted in https://github.com/tensorflow/tensorflow/issues/27303
# Thanks a lot to github.com/rryan for his support!
# Thanks to github.com/padoremu for initially compiling this collection.

# This code has been tested with TF 2.2.0 on Ubuntu 18.04 with Python 3.6.9.


import tensorflow as tf
import numpy as np
import fractions
import math
import param as params


def _dft_matrix_static(dft_length):
  """Calculate the full DFT matrix in numpy."""
  omega = (0 + 1j) * 2.0 * np.pi / float(dft_length)
  # Don't include 1/sqrt(N) scaling, tf.signal.rfft doesn't apply it.
  return np.exp(omega * np.outer(np.arange(dft_length), np.arange(dft_length)))


def _naive_rdft_static(signal_tensor, fft_length):
    """Implement real-input Fourier Transform by matmul."""
    # We are right-multiplying by the DFT matrix, and we are keeping
    # only the first half ("positive frequencies").
    # So discard the second half of rows, but transpose the array for
    # right-multiplication.
    # The DFT matrix is symmetric, so we could have done it more
    # directly, but this reflects our intention better.
    complex_dft_matrix_kept_values = _dft_matrix_static(fft_length)[:(257), :].transpose()
    real_dft_tensor = tf.constant(np.real(complex_dft_matrix_kept_values).astype(np.float32), name='real_dft_matrix')
    imag_dft_tensor = tf.constant(np.imag(complex_dft_matrix_kept_values).astype(np.float32), name='imaginary_dft_matrix')
    signal_frame_length = 400
    half_pad = 56

    
    # Center-padding.
    pad_values = tf.concat([
        tf.zeros([tf.rank(signal_tensor) - 1, 2], tf.int32),
        [[half_pad, 56]]
    ], axis=0)
    
    padded_signal = tf.pad(signal_tensor, pad_values)
    
    result_real_part = tf.matmul(padded_signal, real_dft_tensor)
    result_imag_part = tf.matmul(padded_signal, imag_dft_tensor)
    
    return result_real_part, result_imag_part



def _fixed_frame_static(signal, frame_length, frame_step):
    """tflite-compatible tf.signal.frame for fixed-size input.
    Args:
        signal: Tensor containing signal(s).
        frame_length: Number of samples to put in each frame.
        frame_step: Sample advance between successive frames.
        first_axis: If true, framing is applied to first axis of tensor; otherwise,
        it is applied to last axis.
    Returns:
        A new tensor where the last axis (or first, if first_axis) of input
        signal has been replaced by a (num_frames, frame_length) array of individual
        frames where each frame is drawn frame_step samples after the previous one.
    Raises:
        ValueError: if signal has an undefined axis length.  This routine only
        supports framing of signals whose shape is fixed at graph-build time.
    """
    signal_shape = [10500]
    
    
    length_samples = 10500
    
    
    num_frames = 64
    
    
    outer_dimensions = []
    result_shape = [64,400]
    # Currently tflite's gather only supports axis==0, but that may still
    # work if we want the last of 1 axes.
    gather_axis = 0

    subframe_length = 80  # pylint: disable=deprecated-method
    subframes_per_frame = 5
    subframes_per_hop = 2
    num_subframes = 131

    
    trimmed_input_size = [10480]
    subframe_shape = [131, 80]
    subframes = tf.reshape(
        tf.slice(
            signal,
            begin=np.zeros(len(signal_shape), np.int32),
            size=trimmed_input_size), subframe_shape)

    # frame_selector is a [num_frames, subframes_per_frame] tensor
    # that indexes into the appropriate frame in subframes. For example:
    # [[0, 0, 0, 0], [2, 2, 2, 2], [4, 4, 4, 4]]
    frame_selector = np.reshape(np.arange(num_frames) * subframes_per_hop, [num_frames, 1])

    # subframe_selector is a [num_frames, subframes_per_frame] tensor
    # that indexes into the appropriate subframe within a frame. For example:
    # [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    subframe_selector = np.reshape(np.arange(subframes_per_frame), [1, subframes_per_frame])

    # Adding the 2 selector tensors together produces a [num_frames,
    # subframes_per_frame] tensor of indices to use with tf.gather to select
    # subframes from subframes. We then reshape the inner-most subframes_per_frame
    # dimension to stitch the subframes together into frames. For example:
    # [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]].
    selector = frame_selector + subframe_selector
    frames = tf.reshape(tf.gather(subframes, selector.astype(np.int32), axis=gather_axis), result_shape)
    
    return frames


def _stft_tflite_static(signal, frame_length, frame_step, fft_length):
    """tflite-compatible implementation of tf.signal.stft.
    Compute the short-time Fourier transform of a 1D input while avoiding tf ops
    that are not currently supported in tflite (Rfft, Range, SplitV).
    fft_length must be fixed. A Hann window is of frame_length is always
    applied.
    Since fixed (precomputed) framing must be used, signal.shape[-1] must be a
    specific value (so "?"/None is not supported).
    Args:
        signal: 1D tensor containing the time-domain waveform to be transformed.
        frame_length: int, the number of points in each Fourier frame.
        frame_step: int, the number of samples to advance between successive frames.
        fft_length: int, the size of the Fourier transform to apply.
    Returns:
        Two (num_frames, fft_length) tensors containing the real and imaginary parts
        of the short-time Fourier transform of the input signal.
    """
    # Make the window be shape (1, frame_length) instead of just frame_length
    # in an effort to help the tflite broadcast logic.
    window = tf.reshape(
        tf.constant(
            (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(0, 1.0, 1.0 / frame_length))
            ).astype(np.float32),
            name='window'), [1, frame_length])
    
    framed_signal = _fixed_frame_static(signal, frame_length, frame_step)
    framed_signal *= window
    
    real_spectrogram, imag_spectrogram = _naive_rdft_static(framed_signal, fft_length)
    
    return real_spectrogram, imag_spectrogram


def _stft_magnitude_tflite_static(signals, frame_length, frame_step, fft_length):
    """Calculate spectrogram avoiding tflite incompatible ops."""
    real_stft, imag_stft = _stft_tflite_static(signals, frame_length, frame_step, fft_length)
    stft_magnitude = tf.sqrt(tf.add(real_stft * real_stft, imag_stft * imag_stft), name='magnitude_spectrogram')
    
    return stft_magnitude

window_length_samples = int(round(params.SAMPLE_RATE * params.STFT_WINDOW_SECONDS))
hop_length_samples = int(round(params.SAMPLE_RATE * params.STFT_HOP_SECONDS))
fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
num_spectrogram_bins = fft_length // 2 + 1
def waveform_to_log_mel_spectrogram_static(waveform, params):
    """Compute log mel spectrogram of a 1-D waveform."""
    with tf.name_scope('log_mel_features'):
        # waveform has shape [<# samples>]

        window_length_samples = int(
            round(params.SAMPLE_RATE * params.STFT_WINDOW_SECONDS))
        hop_length_samples = int(
            round(params.SAMPLE_RATE * params.STFT_HOP_SECONDS))
        fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
        num_spectrogram_bins = fft_length // 2 + 1
        magnitude_spectrogram = _stft_magnitude_tflite_static(
            signals=waveform, 
            frame_length=window_length_samples, 
            frame_step=hop_length_samples, 
            fft_length=fft_length)

        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=params.MEL_BANDS,
            num_spectrogram_bins=257,
            sample_rate=params.SAMPLE_RATE,
            lower_edge_hertz=params.MEL_MIN_HZ,
            upper_edge_hertz=params.MEL_MAX_HZ)

        mel_spectrogram = tf.matmul(
            magnitude_spectrogram, linear_to_mel_weight_matrix)
        log_mel_spectrogram = tf.math.log(
            mel_spectrogram + params.LOG_OFFSET)
        # log_mel_spectrogram has shape [<# STFT frames>, MEL_BANDS]

        return log_mel_spectrogram