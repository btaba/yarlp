FROM tensorflow/tensorflow:1.5.0-py3

RUN apt update &&\
 	apt install --yes libsm6 libxext6 libfontconfig1 libxrender1 python3-tk  python-setuptools libffi-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

ADD requirements.txt .

RUN pip install -r requirements.txt

ADD . /yarlp

WORKDIR /yarlp

RUN python3 setup.py install

# Patch gym install for ALE...
WORKDIR `python3 -c "import atari_py; import os; print(os.path.dirname(atari_py.__file__))"`
RUN make

WORKDIR /yarlp

RUN pytest yarlp
