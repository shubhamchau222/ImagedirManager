'''
author : Shubbham Chau 
Date : 30-oct-2021
'''


from src.utils.all_utils import create_dir , genFilesnamepkl , load_config
from sklearn.metrics.pairwise import cosine_similarity
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from mtcnn import MTCNN
from PIL import Image
import numpy as np
import argparse
import pickle
import yaml 
import cv2
import os 





model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')


def detectANDpred(confg1 , confg2 , threshould=0.7):
    '''
        Fun: detectAndpred
            i/p:
                confg1 : structure-config File ( .yaml)
                confg2 : pred-config File ( .yaml)
                confg1 : threshould = 0.7 -> int (changable parameter)      
    '''
    conf1 = load_config(confg1)
    pred_confg = load_config(confg2)
    predImgpath = pred_confg['Prediction']['pred_imgPath']
    # detect the faces from imgs

    detector = MTCNN()
    predImg = cv2.imread(predImgpath)
    detectionOp = detector.detect_faces(predImg)
    x,y,width,height = detectionOp[0]['box']         # (  x,y , width , height)  => x2,y2(diagonal_point) (x+width , y+height)
    x1,y1,x2,y2 = x,y,x+width,y+height

    # create the boxes and crop the img
    detected_face = predImg[y1:y2,x1:x2]
    #  extract its features
    image = Image.fromarray(detected_face)
    image = image.resize((224,224))
    face_array = np.asarray(image)
    face_array = face_array.astype('float32')    
    expanded_img = np.expand_dims(face_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()

    # let's bring the feature list here
    maindir = conf1['internal_ops']['artifact_dir']
    imgfeaturesdir = conf1['internal_ops']['ExtractedFeatures']['imgfeaturesdir']  # embeddingImgdata
    imgFeaturesFilename = conf1['internal_ops']['ExtractedFeatures']['imgFeaturesFilename']  # features_embedding.pkl 
    Filename = os.path.join(maindir , imgfeaturesdir , imgFeaturesFilename)
    # embeddings 
    embeddings = pickle.load(open(file=Filename , mode='rb'))
    similar_imgs_score = []
    for i in range(len(embeddings)):
        similar_imgs_score.append(cosine_similarity(result.reshape(1,-1),embeddings[i].reshape(1,-1))[0][0])

    # we'll get list of similarity with all imgs [0.20 , 0.1 , ...] [f1,f2,.....] sequentially....

# index_pos = sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1])[0][0]

# temp_img = cv2.imread(filenames[index_pos])
# cv2.imshow('output',temp_img)
# cv2.waitKey(0)
# # recommend that image




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config1" ,"-c1" , default='configs/structure.yaml' ) 
    args.add_argument("--config2" ,"-c2" , default='configs/pred_config.yaml' ) 
    parsed_args = args.parse_args()
    try:
        detectANDpred(parsed_args.config1 , parsed_args.config2)
    except Exception as e:
        raise e 
    