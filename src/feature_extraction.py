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

def createfilenamepkl(config_data:dict) -> list:
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
    return [messedDirPath , pklfilepath]

  
vggf_model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

# extract the features from img (sub-function)

def get_features(img_path , model):
    ''' load the img => cnvt-array => chng dims => preprocess => extract
        =>Features with the help of model 
    '''
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

def feature_extractor(config_data:dict , file_names_pklfile_path:str):   # require the filename also 
    config_data=config_data
    extracted_features = []
    try:
        if os.path.isfile(file_names_pklfile_path):
            filenames = pickle.load(open(file_names_pklfile_path , mode='rb'))
            for file in filenames:
                feature = get_features(img_path=file , model=vggf_model)
                extracted_features.append(feature)    

            # save extracted features at given loc 
            maindir = config_data['internal_ops']['artifact_dir']
            imgfeaturesdir = config_data['internal_ops']['ExtractedFeatures']['imgfeaturesdir']
            imgFeaturesFilename = config_data['internal_ops']['ExtractedFeatures']['imgFeaturesFilename']
            imgfeaturesdir_full_path = os.path.join(maindir,imgfeaturesdir)
            Filename = os.path.join(maindir , imgfeaturesdir , imgFeaturesFilename)      # features_embedding.pkl

            # check imgfeaturesdir is present/not else:create
            create_dir([imgfeaturesdir_full_path]) 
            pickle.dump(extracted_features,open(Filename , 'rb'))

        else:
            print('Filename Pickle file not Found....')      
      
    except Exception as e :
        raise e 

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--configdata" ,"-d" , default=structure_data ) 
    parsed_args = args.parse_args()
    try:
        OPFN = createfilenamepkl(parsed_args.configdata)
        FnamePklPath = OPFN[1] 
        feature_extractor(config_data=parsed_args.configdata , File_names_pklfile_path=FnamePklPath)
        print('Feature Extaractor Done successFully....')

    except Exception as e:
        raise e 


   
