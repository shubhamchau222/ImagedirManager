from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm
import argparse


model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

# messed_up_dir => get-images/there_names => 




# extracting the features from the img 
def feature_extracor(img_path,model):
    pass 

