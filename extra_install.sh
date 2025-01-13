pip install torch==2.1.0 torchvision==0.16.0 -f https://download.pytorch.org/whl/cu12.4
pip install git+https://github.com/facebookresearch/pytorch3d.git

## If pytorch3d can't find cuda binary, find (install) cuda and execute the line below 
##export CUDA_HOME=/usr/local/cuda-12.4; export PATH=$CUDA_HOME/bin:$PATH; export  LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
