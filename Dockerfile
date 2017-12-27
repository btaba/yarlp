FROM quay.io/travisci/ci-python:packer-1475058170

#USER travis

ADD . /yarlp
#ADD ./external /yarlp/external
#ADD setup.py /yarlp/setup.py

WORKDIR /yarlp

RUN /home/travis/virtualenv/python3.5/bin/pip install -r requirements.txt

RUN /home/travis/virtualenv/python3.5/bin/python setup.py install

#RUN /home/travis/virtualenv/python3.5/bin/pytest
