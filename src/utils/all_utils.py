import yaml 
import os
import pickle 


def create_dir(dirs_paths:list)-> None:
    for dir_path in dirs_paths:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path , exist_ok= True) 



# load config file and return all the data inside this s
def load_config(config_filePath):
    try:
        if os.path.isfile(config_filePath):
            print('config file is present , reading config....')
            with open(config_filePath , mode='r') as f:
                data=yaml.safe_load(f)
                f.close()
                return data 
        else:
            print('config file not detected....')
    except Exception as e:
        print(f'Error in load config fuction {str(e)}')

        


# this fun will extract the names/path of images & dumped pkl at given loc
def genFilesnamepkl(messedDirPath:str , dumpingPath:str):
    filenames=[]
    try:
        if os.path.isdir(messedDirPath):
            for imgFile in os.listdir(messedDirPath):
                if (imgFile.endswith('jpg') or imgFile.endswith('png')):
                    filenames.append(imgFile)
                else:
                    pass                        
            # dumped the filename into pickle format
            pickle.dump(filenames , open(dumpingPath,'wb'))
            return messedDirPath       
        else:
            pass
    except Exception as e:
        print('error in genfilenamepkl')
        raise e 





