from tqdm import tqdm
import requests
import math
import os
import tarfile
import numpy as np
import pandas as pd

import audioTools

GSCmdCats = {
    'unknown': 0,
    'silence': 0,
    '_unknown_': 0,
    '_silence_': 0,
    '_background_noise_': 0,
    'yes': 2,
    'no': 3,
    'up': 4,
    'down': 5,
    'left': 6,
    'right': 7,
    'on': 8,
    'off': 9,
    'stop': 10,
    'go': 11,
    'zero': 12,
    'one': 13,
    'two': 14,
    'three': 15,
    'four': 16,
    'five': 17,
    'six': 18,
    'seven': 19,
    'eight': 20,
    'nine': 1}

numGSCmdCats = 21


def SetUpGoogleSpeechCmd(forceDownload=False):
    """
    Prepares Google Speech commands dataset version 2 for use

    Returns full path to training, validation and test file list and file categories
    """

    downloadGoogleSpeechDB(forceDownload)
    basePath = 'sd_GSCmd'

    GSCmdCats = {
        'unknown': 0,
        'silence': 0,
        '_unknown_': 0,
        '_silence_': 0,
        '_background_noise_': 0,
        'yes': 2,
        'no': 3,
        'up': 4,
        'down': 5,
        'left': 6,
        'right': 7,
        'on': 8,
        'off': 9,
        'stop': 10,
        'go': 11,
        'zero': 12,
        'one': 13,
        'two': 14,
        'three': 15,
        'four': 16,
        'five': 17,
        'six': 18,
        'seven': 19,
        'eight': 20,
        'nine': 1,
        'backward': 21,
        'bed': 22,
        'bird': 23,
        'cat': 24,
        'dog': 25,
        'follow': 26,
        'forward': 27,
        'happy': 28,
        'house': 29,
        'learn': 30,
        'marvin': 31,
        'sheila': 32,
        'tree': 33,
        'visual': 34,
        'wow': 35
    }
    
    numGSCmdCats = 36


    print('Converting test set WAVs to numpy files')
    audioTools.WAV2Numpy(basePath + '/test/')
    print('Converting training set WAVs to numpy files')
    audioTools.WAV2Numpy(basePath + '/train/')

    # read split from files and all files in folders
    testWAVs = pd.read_csv(basePath + '/train/testing_list.txt', sep=" ", header=None)[0].tolist()
    valWAVs = pd.read_csv(basePath + '/train/validation_list.txt', sep=" ", header=None)[0].tolist()

    testWAVs = [os.path.join(basePath + '/train/', f + '.npy') for f in testWAVs if f.endswith('.wav')]
    valWAVs = [os.path.join(basePath + '/train/', f + '.npy') for f in valWAVs if f.endswith('.wav')]
    allWAVs = []

    for root, dirs, files in os.walk(basePath + '/train/'):
        allWAVs += [root + '/' + f for f in files if f.endswith('.wav.npy')]

    trainWAVs = list(set(allWAVs) - set(valWAVs) - set(testWAVs))

    testWAVsREAL = []

    for root, dirs, files in os.walk(basePath + '/test/'):
        testWAVsREAL += [root + '/' + f for f in files if f.endswith('.wav.npy')]

    # get categories
    testWAVlabels = [getFileDir(f, GSCmdCats) for f in testWAVs]
    valWAVlabels = [getFileDir(f, GSCmdCats) for f in valWAVs]
    trainWAVlabels = [getFileDir(f, GSCmdCats) for f in trainWAVs]
    testWAVREALlabels = [getFileDir(f, GSCmdCats)
                         for f in testWAVsREAL]

    # background noise should be used for validation as well
    backNoiseFiles = [trainWAVs[i] for i in range(len(trainWAVlabels))
                      if trainWAVlabels[i] == GSCmdCats['silence']]
    backNoiseCats = [GSCmdCats['silence']
                     for i in range(len(backNoiseFiles))]
    if numGSCmdCats == 12:
        valWAVs += backNoiseFiles
        valWAVlabels += backNoiseCats

    # build dictionaries
    testWAVlabelsDict = dict(zip(testWAVs, testWAVlabels))
    valWAVlabelsDict = dict(zip(valWAVs, valWAVlabels))
    trainWAVlabelsDict = dict(zip(trainWAVs, trainWAVlabels))
    testWAVREALlabelsDict = dict(zip(testWAVsREAL, testWAVREALlabels))


    # info dictionary
    trainInfo = {'files': trainWAVs, 'labels': trainWAVlabelsDict}
    valInfo = {'files': valWAVs, 'labels': valWAVlabelsDict}
    testInfo = {'files': testWAVs, 'labels': testWAVlabelsDict}
    testREALInfo = {'files': testWAVsREAL, 'labels': testWAVREALlabelsDict}
    gscInfo = {'train': trainInfo,
               'test': testInfo,
               'val': valInfo,
               'testREAL': testREALInfo}

    print('GSCmd is set up')

    return gscInfo, numGSCmdCats


def getFileDir(file, catDict):
    """
    Receives a file with name sd_GSCmd/train/<cat>/<filename> and returns an integer that is catDict[cat]
    """
    categ = os.path.basename(os.path.dirname(file))
    return catDict.get(categ, 0)


def downloadGoogleSpeechDB(forceDownload=False):
    """
    Downloads Google Speech commands dataset
    """

    if os.path.isdir("sd_GSCmd/") and not forceDownload:
        print('GSCmds is already downloaded')
    else:
        if not os.path.exists("sd_GSCmd/"):
            os.makedirs("sd_GSCmd/")
        trainFiles = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
        testFiles = 'http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz'
        downloadFile(testFiles, 'sd_GSCmd/test.tar.gz')
        downloadFile(trainFiles, 'sd_GSCmd/train.tar.gz')

    # extract files
    if not os.path.isdir("sd_GSCmd/test/"):
        extract('sd_GSCmd/test.tar.gz', 'sd_GSCmd/test/')

    if not os.path.isdir("sd_GSCmd/train/"):
        extract('sd_GSCmd/train.tar.gz', 'sd_GSCmd/train/')



def downloadFile(url, fName):
    
    """
    Downloads the file from url to file fName
    """

    r = requests.get(url, stream=True)

    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0
    print('Downloading {} into {}'.format(url, fName))
    with open(fName, 'wb') as f:
        for data in tqdm(r.iter_content(block_size),
                         total=math.ceil(total_size // block_size),
                         unit='KB',
                         unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")


def extract(fname, folder):
    
    """
    Extracts tar.gz file
    """
    
    print('Extracting {} into {}'.format(fname, folder))
    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall(path=folder)
        tar.close()
    elif (fname.endswith("tar")):
        tar = tarfile.open(fname, "r:")
        tar.extractall(path=folder)
        tar.close()
