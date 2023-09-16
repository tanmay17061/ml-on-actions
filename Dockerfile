ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
ARG PY_VER=3.9
ARG UBUNTU_VER=20.04

FROM ubuntu:${UBUNTU_VER}

RUN apt-get update && apt-get install -yq curl wget

# Install miniconda to /miniconda
ARG CONDA_VER
ARG OS_TYPE

RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
RUN bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b
RUN rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

ARG PY_VER

RUN conda install -c anaconda -y python=${PY_VER} && conda init bash
RUN conda install git pip poetry evidently
RUN conda info && which pip && which poetry

ADD . repository/

RUN cd repository && poetry install

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
RUN cd repository && poetry run dvc pull data

CMD ["/bin/sh", "-ec", "cd repository/src && poetry run uvicorn http_server.main:app --host 0.0.0.0 --port 8000"]

