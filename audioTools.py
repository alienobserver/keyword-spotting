import os
import itertools
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt


def plotConfusionMatrix(confusionMatrix, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Greens):

    """
    Plot the confusion matrix, option of printing the normalized one can be 
    applied by choosing 'normalize = True' arg while calling
    """

    # checking to plot normalized or not
    if normalize:
        confusionMatrix = confusionMatrix.astype('float') / confusionMatrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    plt.figure(figsize=(15, 15))
    plt.imshow(confusionMatrix, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.3f' if normalize else 'd'
    thresh = confusionMatrix.max() / 2.
    for i, j in itertools.product(range(confusionMatrix.shape[0]), range(confusionMatrix.shape[1])):
        plt.text(j, i, format(confusionMatrix[i, j], fmt), size=11,
                 horizontalalignment="center",
                 color="white" if confusionMatrix[i, j] > thresh else "black")


    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)
    plt.savefig('picConfMatrix.png', dpi=400)
    plt.tight_layout()


def WAV2Numpy(folder, sr=None):

    """
    Converts given WAV's from 'folder' to numpy arrays, deletes
    them afterwards
    """

    allFiles = []
    for root, dirs, files in os.walk(folder):
        allFiles += [os.path.join(root, f) for f in files
                     if f.endswith('.wav')]

    for file in tqdm(allFiles):
        x = tf.io.read_file(str(file))
        y, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)

        # if we want to write the file later
        np.save(file + '.npy', y.numpy())
        os.remove(file)

def specFn(input_signal, nfft, window, stride, name=None):

    """
    Create spectrogram

    Params:
      input: An 1-D audio signal Tensor.
      nfft: Size of Fast Fourier Transform.
      window: Size of window.
      stride: Size of hops between windows.
      name: A name for the operation (optional).

    Return:
      A tensor of spectrogram.
    """

    return tf.math.abs(
        tf.signal.stft(
            input_signal,
            frame_length=window,
            frame_step=stride,
            fft_length=nfft,
            window_fn=tf.signal.hann_window,
            pad_end=True,
        )
    )

def normalizedMelSpec(x, sr=16000, n_mel_bins=80):

    """
    Normalizes given mel spectogram
    """

    spec_stride = 128
    spec_len = 1024

    spectrogram = specFn(
        x, nfft=spec_len, window=spec_len, stride=spec_stride
    )

    num_spectrogram_bins = spec_len // 2 + 1 
    lower_edge_hertz, upper_edge_hertz = 40.0, 8000.0
    num_mel_bins = n_mel_bins
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sr, lower_edge_hertz,
      upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # get log magnitude mel scale specs
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    avg = tf.math.reduce_mean(log_mel_spectrograms)
    std = tf.math.reduce_std(log_mel_spectrograms)

    return (log_mel_spectrograms - avg) / std

def plotRawWav(audio, imgHeight):
    
    """
    Plots Raw Waveform of a audio
    """

    plt.figure(figsize=(17,imgHeight))
    plt.title('Raw waveform', fontsize=30)
    plt.ylabel('Amplitude', fontsize=30)
    plt.xlabel('Sample index', fontsize=30)
    plt.plot(audio)
    plt.show()


def plotAttW(attW, imgHeight):
    
    """
    Plots the weights of the attention layer
    """

    plt.figure(figsize=(17,imgHeight))
    plt.title('Attention weights (log)', fontsize=30)
    plt.ylabel('Log of attention weight', fontsize=30)
    plt.xlabel('Mel-spectrogram index', fontsize=30)
    plt.plot(np.log(attW))
    plt.show()

def plotSpecVis(specs, imgHeight):
   
    """
    Visulaizes the spectogram
    """
    
    plt.figure(figsize=(17,imgHeight*2))
    plt.pcolormesh(specs)
    plt.title('Spectrogram visualization', fontsize=30)
    plt.ylabel('Frequency', fontsize=30)
    plt.xlabel('Time', fontsize=30)
    plt.show()


def plotCatAcc(results):
    
    """
    Plot the history of sparse categorical accuracy
    """
    
    plt.plot(results.history['sparse_categorical_accuracy'])
    plt.plot(results.history['val_sparse_categorical_accuracy'])
    plt.title('Categorical accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plotLoss(results):
    
    """
    Plot the loss history
    """

    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    