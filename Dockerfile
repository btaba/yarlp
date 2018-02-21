FROM tensorflow/tensorflow:1.5.0-py3

RUN apt update &&\
 	apt install --yes libsm6 libxext6 libfontconfig1 libxrender1 python3-tk  python-setuptools libffi-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

ADD requirements.txt .

RUN pip install -r requirements.txt

ADD . /yarlp

WORKDIR /yarlp

RUN python3 setup.py install

# Patch gym install for ALE...
WORKDIR /usr/local/lib/python3.5/dist-packages/atari_py-0.1.1-py3.5.egg/atari_py/ale_interface/
RUN make

WORKDIR /yarlp

RUN pytest yarlp
