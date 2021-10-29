from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import os
import yaml 
from src.utils.all_utils import create_dir , genFilesnamepkl , load_config

# load& read configs 

structure_data = load_config("configs/structure.yaml" )
model_data = None 

def createfilenamepkl(config_data:dict) -> str:
    data = config_data
    main_dir = data['internal_ops']['artifact_dir']
    dumpdir = data['internal_ops']['dumpeddir']
    pklname=data['internal_ops']['src_pkl_filename']
    print(type(pklname))
    messeddirPath= data['internal_ops']['messed_dirPath']
    dumdirFullpath= os.path.join(main_dir,dumpdir)
    pklfilepath= os.path.join(dumdirFullpath,pklname)
    # print(pklfilepath)
    # first create dir to store pkl filenames
    create_dir([main_dir])  # check if artifacts dir is available ? if no then create else:pass    
    create_dir([dumdirFullpath])  # creting dir to store pkl files 
    messedDirPath = genFilesnamepkl(messedDirPath=messeddirPath , dumpingPath=pklfilepath) 
    return messedDirPath

  
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

# extract the features from img (sub-function)

def get_features(img_path , model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result






if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--configdata" ,"-d" , default=structure_data ) 
    parsed_args = args.parse_args()
    createfilenamepkl(parsed_args.configdata)
