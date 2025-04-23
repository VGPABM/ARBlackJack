import os
from keras.models import load_model
import cv2
import numpy as np
from keras.layers import Input, Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Model
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import expand_dims
from keras.utils import load_img
from keras.utils import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

import ModulKlasifikasiCitraCNN as MCNN

LabelKelas=("2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "A",
            "J",
            "K",
            "Q",
            )

# LabelKelas=("club",
#             "diamond",
#             "heart",
#             "spade",
# )

DirektoriDataSet="datasetnew/rank"

JumlahEpoh = 25;

FileBobot = "bobotRank.h5"
ModelCNN,history = MCNN.TrainingCNN(JumlahEpoh,DirektoriDataSet,LabelKelas ,FileBobot)
ModelCNN.summary()