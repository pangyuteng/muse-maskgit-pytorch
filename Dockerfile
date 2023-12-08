FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
RUN pip install muse-maskgit-pytorch
# COPY . /opt/muse-maskgit-pytorch
# WORKDIR /opt/muse-maskgit-pytorch
# RUN python3 setup.py install

# RUN useradd -r 
# USER app