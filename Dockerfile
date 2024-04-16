FROM ubuntu:22.04
COPY . /t-core
WORKDIR /t-core

RUN apt update
RUN apt install --assume-yes --no-install-recommends build-essential python3-dev libopenblas-dev python3-pip
RUN pip3 install -r requirements.txt
