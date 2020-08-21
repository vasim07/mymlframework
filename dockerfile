# docker pull continuumio/miniconda3:4.8.2
# git clone 
# docker build --tag mlframe .

# docker run -it -v F:/Vasim/PythonStuff/ml-framework/cdmlfw/input:/home/mlframework/input --publish 5000:5000 --publish 5555:5555 mlframe /bin/bash

FROM continuumio/miniconda3
LABEL maintainer="Vasim"
LABEL version=0.1
LABEL description="ML Framework try"

RUN apt-get update && \
      apt-get -y install sudo

RUN apt-get -y install gcc python3-dev

RUN sudo apt install -y redis-server

# TODO
# RUN useradd -m docker && echo "vasim:vasim" | chpasswd && adduser docker sudo
# RUN useradd -m docker && echo "vasim:vasim" | chpasswd && adduser celery sudo

RUN mkdir mlframework
WORKDIR /home/mlframework

ADD . .
# RUN rm -r input
RUN rm var.sh

RUN pip install -r ./requirements.txt

# Need refresh
VOLUME F:/Vasim/PythonStuff/ml-framework/cdmlfw/input:mlframework/input
COPY ./celeryd /etc/defaults
COPY ./redis.conf /etc/redis/redis.conf

RUN service redis-server start
RUN /etc/init.d/celeryd start

EXPOSE 5000:5000
EXPOSE 5555:5555

# USER docker
CMD bin/bash