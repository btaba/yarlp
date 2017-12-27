FROM tensorflow/tensorflow:1.4.0-py3

RUN apt update &&\
 	apt install --yes libsm6 libxext6 libfontconfig1 libxrender1 python3-tk libffi-dev

ADD requirements.txt .

RUN pip install -r requirements.txt

ADD . /yarlp

WORKDIR /yarlp

RUN python setup.py install

RUN pytest yarlp
