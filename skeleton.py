import os 


def create_skeleton():
    dirs = ['artifacts' , 'artifacts/model' , 'artifacts/data'  ,
                'src', 'src/utils']
    for dir in dirs:
        os.makedirs(os.path.join(os.getcwd() , dir) ,exist_ok=True )

    
if __name__ == '__main__':
    create_skeleton()


