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

def createfilenamepkl(configPath:str) -> str:
    data = load_config(configPath)
    print(data)
    main_dir = data['internal_ops']['artifact_dir']
    dumpdir = data['internal_ops']['dumpeddir']
    pklname=data['internal_ops']['src_pkl_filename']
    messeddirPath= data['internal_ops']['messed_dirPath']
    dumdirFullpath= os.path.join(main_dir,dumpdir)
    pklfilepath= os.path.join(dumdirFullpath,pklname)
    # print(pklfilepath)

    # first create dir to store pkl filenames
    create_dir([main_dir])  # check if artifacts dir is available ? if no then create else:pass    
    create_dir([dumdirFullpath])  # creting dir to store pkl files 
    messedDirPath = genFilesnamepkl(messedDirPath=messeddirPath , dumpingPath=pklfilepath) 
    return messedDirPath

  
# model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

print('Hello world')


# def feature_extracor(img_path,model):
#     pass 


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config" ,"-c" , default="configs/structure.yaml" ) 
    # adding the arguments , default = 'congig filepath'
    parsed_args = args.parse_args()
    print(parsed_args)
    createfilenamepkl(parsed_args.config)
