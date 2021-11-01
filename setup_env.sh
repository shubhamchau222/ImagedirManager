#!/usr/bin/env bash


conda create -n imgdmanager python==3.7 -y 
source activate imgdmanager
conda env list 
# to confirm the activated env 


# install the dependancies 

pip install -r requirements.txt 

pip list   # check all dependancies are installed or not 

# check local packages are installed or not 





# to run this file hit command ($ bash setup_env.sh )