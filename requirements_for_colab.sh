#!/bin/sh
pip install tensorflow==1.14
apt install swig
cd tf_pose/pafprocess && swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
cd ../..
#ls
#pwd
pip3 install -r requirements.txt && python setup.py install
