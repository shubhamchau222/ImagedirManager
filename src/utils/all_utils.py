from numpy.lib.utils import source
import yaml 
import os
import pickle 
import shutil 


def create_dir(dirs_paths:list)-> None:
    for dir_path in dirs_paths:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path , exist_ok= True) 
    # print(f'{dirs_paths} dir created')



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
        raise e 

        


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


def move_file_to_dir(imgfilepath , destdir):
    '''
        Method: move_file_to_dir
        inputs: 
                imgfilepath :  imgpath which want to move : str
                destdir : In which dir you want to move : str
        returns : None 
    '''
    try:
        if os.path.isdir(destdir):
            if os.path.isfile(imgfilepath):
                shutil.move(destdir , imgfilepath)
            else:
                pass
        else:
            print(f'dir {destdir} is not Found..')   
           
    except Exception as e:
        print(f'Error occures in fun : move_file_to_dir.. {str(e)} ')
        raise e 





