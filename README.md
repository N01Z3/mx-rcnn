To run this ficking code:

for anaconda2

git clone --recursive https://github.com/dmlc/mxnet
git checkout v0.9.5
git submodule update

make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda-8.0 USE_CUDNN=1 EXTRA_OPERATORS = ~/mxnet/example/rcnn/operator

cd python/

sudo ~/anaconda2/bin/python setup.py install