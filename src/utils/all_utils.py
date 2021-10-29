import yaml 
import os 



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




