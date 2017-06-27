import os
import sys
import random
import numpy as np
import csv
from ..logger import logger
from ..config import default


def getImageLst():
    if not os.path.exists(default.train_lst) or os.path.exists(default.valid_lst):
        makeImageLst(inputFilePath = default.train_list, dataDir = default.data_dir, 
             labelFile = default.label_list, fileType = 'train')
        logger.info('create train.lst and valid.lst')
    if not os.path.exists(default.test_lst):
        makeImageLst(inputFilePath = default.train_list, dataDir = default.data_dir, 
             labelFile = default.label_list, fileType = 'train')
        logger.info('create test.lst')

    return

def makeImageLst(inputFilePath, dataDir, labelFile, fileType = 'train'):
    if fileType == 'train':
        trainFile = extractDir(inputFilePath) + '/train.lst'
        validFile = extractDir(inputFilePath) + '/valid.lst'
    elif fileType == 'test':
        testFile = extractDir(inputFilePath) + '/test.lst'
         
    dataFileList, inputFileLabelNames = readInputFile(inputFilePath, dataDir)
    labelDict = readLabelFile(labelFile)
    
    # given labelNames and labelDict, retrieve labels for each item dataFileList
    labels = map(lambda x: labelDict[x], inputFileLabelNames)
    writeToFileWrapper(inputFilePath, dataFileList, labels, fileType)
    return
        
def writeToFileWrapper(inputFilePath, dataFileList, labels, fileType):
    if fileType == 'train':
        trainFile = extractDir(inputFilePath) + '/train.lst'
        validFile = extractDir(inputFilePath) + '/valid.lst'
        trainIndices, validIndices = trainValidSplit(len(dataFileList), rate = 0.8)
        writeToFile(trainFile, trainIndices, dataFileList, labels)
        writeToFile(validFile, validIndices, dataFileList, labels)
    elif fileType == 'test':
        testFile = extractDir(inputFilePath) + '/test.lst'
        testIndices = range(len(dataFileList))
        writeToFile(testFile, testIndices, dataFileList, labels)
    return

def writeToFile(fileName, indices, dataFileList, labels):
    fileHandler = csv.writer(open(fileName, "w"), delimiter='\t', lineterminator='\n')
    image_list = []
    counter = 0
    for i in indices:
        for img in os.listdir(dataFileList[i]):
            image_list.append((counter, labels[i], dataFileList[i] + '/' + img))
            counter += 1
    
    for il in image_list:
        fileHandler.writerow(il)
    return 
            
def trainValidSplit(num, rate = 0.8):
    """
    Given a split rate, rate of all nums is training indices, rest is valid indices
    """
    trainSize = int(num * rate)
    trainIndices = sorted(np.random.choice(num, size = trainSize, replace = False).tolist())
    validIndices = []
    trainIndex = 0
    for i in xrange(num):
        if trainIndex >= len(trainIndices) or i != trainIndices[trainIndex]:
            validIndices.append(i)
        else:
            trainIndex += 1
    return trainIndices, validIndices
        
def extractDir(inputFilePath):
    """
    Get the directory of the inputFile, 
    for example:'ucf/list01.txt'=> 'ucf'
    """
    return '/'.join((inputFilePath.split('/')[:-1]))

def readLabelFile(labelFile):
    """
    read the label document and return a dict{VideoName: label}
    The data format in the labelFile should be like:
    '2 ApplyLipstick'
    """
    labelFile = open(labelFile, 'rU')
    labelDict = {}
    for line in labelFile:
        line_list = line.replace('\n', '').split(' ')
        labelDict[line_list[1]] = int(line_list[0])
    return labelDict

def readInputFile(inputFilePath, dataDir):
    """
    The data format in the inputFile should be like:
    'ApplyEyeMakeup/v_ApplyEyeMakeup_g10_c02.avi 1'
    """
    inputFile = open(inputFilePath, 'rU')
    inputFileInfo = filter(lambda x: x.find('.avi') != -1, inputFile)
    # process conflict in InputFileInfo
    inputFileInfo = map(lambda x: x.replace('HandStandPushups', 'HandstandPushups'), inputFileInfo)
    dataFileList = map(lambda x: dataDir  + (x.split('.')[0]).split('/')[-1],
                           inputFileInfo)
    inputFileLabelNames = map(lambda x: (x.split('.')[0]).split('/')[-2],
                           inputFileInfo)
    return dataFileList, inputFileLabelNames
