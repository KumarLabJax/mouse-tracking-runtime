FROM gcr.io/cloud-ml-public/training/pytorch-gpu.1-7

RUN pip install \
EasyDict \
shapely \
Cython \
json_tricks \
scikit-image \
yacs \
imageio-ffmpeg \
h5py \
tensorboard \
opencv-python


RUN mkdir -p /deep-hres-net
COPY . /deep-hres-net

WORKDIR /deep-hres-net
RUN pip install -r requirements.txt
