FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN mkdir /.cache && chmod -R 777 /.cache
