# System
FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

# Installing Dependancies
RUN apt update
RUN apt-get clean

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y wget python3-pip unzip

RUN python3 -m pip install pip --upgrade

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install pip requirements
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

COPY . .
