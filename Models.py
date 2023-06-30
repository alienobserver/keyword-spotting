import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers as L
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
import math
import audioTools

def getMelspecModel(iLen=None):

    """
    returns the Mel Spectogram
    """

    inp = L.Input((iLen,), name='input')
    mel_spec = audioTools.normalizedMelSpec(inp)
    melspecModel = Model(inputs=inp, outputs=mel_spec, name='normalized_spectrogram_model')
    return melspecModel

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.4
    epochs_drop = 15.0
    lrate = initial_lrate * math.pow(drop,  
            math.floor((1+epoch)/epochs_drop))
    
    if (lrate < 4e-5):
        lrate = 4e-5
      
    print('Changing learning rate to {}'.format(lrate))
    return lrate

def AttRNNModel(nCategories, samplingrate=16000, inputLength=16000, rnn_func=L.LSTM):

    """
    Attention Recurrent Neural Network for keyword spotting
    """

    # simple LSTM
    sr = samplingrate
    iLen = inputLength

    inputs = L.Input((inputLength,), name='input')

    m =  getMelspecModel(iLen=inputLength)
    m.trainable = False

    x = m(inputs)
    x = tf.expand_dims(x, axis=-1, name='mel_stft')

    x = L.Conv2D(10, (5, 1), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)

    x = L.Conv2D(1, (5, 1), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)

    x = L.Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim')(x)

    x = L.Bidirectional(rnn_func(64, return_sequences=True))(x)
    x = L.Bidirectional(rnn_func(64, return_sequences=True))(x)

    xFirst = L.Lambda(lambda q: q[:, -1])(x)  
    query = L.Dense(128)(xFirst)

    # attention
    attScores = L.Dot(axes=[1, 2])([query, x])
    attScores = L.Softmax(name='attSoftmax')(attScores) 

    # rescale
    attVector = L.Dot(axes=[1, 1])([attScores, x]) 

    x = L.Dense(64, activation='relu')(attVector)
    x = L.Dense(32)(x)

    output = L.Dense(nCategories, activation='softmax', name='output')(x)

    model = Model(inputs=[inputs], outputs=[output])

    return model

def createModel(nCategs, sr):

    """
    Create a model based on Attention RNN 
    """

    model = AttRNNModel(nCategs, samplingrate = sr, inputLength = None)
    model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy'])

    return model

def trainAndSave(nCategs, sr, trainGen, valGen, model, name = 'attRNN-weights.h5'):

    """
    Function which trains a Attention RNN model on given train and val data
    then saves the weights
    """

    lrate = LearningRateScheduler(step_decay)

    earlystopper = EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=10,
                                verbose=1, restore_best_weights=True)
    checkpointer = ModelCheckpoint(name, monitor='val_sparse_categorical_accuracy', verbose=1, save_best_only=True)

    results = model.fit(trainGen, validation_data=valGen, epochs=60, use_multiprocessing=False, workers=4, verbose=1,
                        callbacks=[earlystopper, checkpointer, lrate])

    model.save(name)

    return model, results