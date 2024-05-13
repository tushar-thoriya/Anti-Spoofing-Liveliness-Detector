import os
import random
import shutil
from itertools import islice

outputFolderPath = "Dataset/SplitData"
inputFolderPath = "Dataset/all"

splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}


try:
    shutil.rmtree(outputFolderPath)
    print("Remvoed Directory")
except OSError as e:
    os.mkdir(outputFolderPath)
    

## --------- Directories to create -------------
os.makedirs(f"{outputFolderPath}/train/images", exist_ok= True)
os.makedirs(f"{outputFolderPath}/train/labels", exist_ok= True)
os.makedirs(f"{outputFolderPath}/val/images", exist_ok= True)
os.makedirs(f"{outputFolderPath}/val/labels", exist_ok= True)
os.makedirs(f"{outputFolderPath}/test/images", exist_ok= True)
os.makedirs(f"{outputFolderPath}/test/labels", exist_ok= True)


## --------- Get the names -------------
listNames = os.listdir(inputFolderPath)
# print(listNames)
# print(len(listNames))

uniquNames = set()
for name in listNames:
    uniquNames.add(name.split('.')[0])

uniquNames = list(uniquNames)
# print(uniquNames)
# print(len(uniquNames))

## --------- Shuffle -------------
random.shuffle(uniquNames)


## --------- Find the number of images for each folder -------------
lenData = len(uniquNames)
print(f"Total Images: {lenData}")

lenTrain = int(lenData * splitRatio['train'])
lenVal = int(lenData * splitRatio['val'])
lenTest = int(lenData * splitRatio['test'])

## --------- copy the files -------------