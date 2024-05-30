FROM python:3.10

RUN apt-get update && apt-get install -y \
    gcc\
    python3-pip \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-setuptools \
    git \
    python3-opencv \
    python3-numpy \
    nano \
    wget 


RUN pip3 install --upgrade pip

RUN pip3 install --no-cache-dir \
    pynacl \
    cryptography \
    pyOpenSSL \
    cffi \
    numpy


COPY converter.py /


RUN wget https://github.com/ViktorPavlovA/rknn-toolkit2-fork-1.5.2/raw/v1.5.2/packages/rknn_toolkit2-1.5.2+b642f30c-cp310-cp310-linux_x86_64.whl

RUN pip3 install --no-cache-dir rknn_toolkit2-1.5.2+b642f30c-cp310-cp310-linux_x86_64.whl

RUN git clone https://github.com/airockchip/ultralytics_yolov8

RUN pip3 install --no-cache-dir -r ultralytics_yolov8/requirements.txt

RUN cd ultralytics_yolov8 && python3 setup.py install

# RUN rm -rf ultralytics_yolov8

RUN git clone https://github.com/airockchip/rknn_model_zoo.git


RUN mkdir /pt/
ADD COCO/ /
# # open port
EXPOSE 5000

CMD ["bash"]