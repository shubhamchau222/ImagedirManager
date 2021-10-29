from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm
import argparse 


model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')


def feature_extracor():
    pass 

