# hash:sha256:9fc7d46102b301952f2e86740f5e5f0f258cee6ac29cdbe6dee1e36d94768566
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-linuxlibertine \
    && rm -rf /var/lib/apt/lists/*
    
RUN conda install -y --channel=conda-forge mamba

RUN mamba install -y --channel=conda-forge \
        cloudpickle==2.2.1 \
        dill==0.3.6 \
        fastparquet=2023.2.0 \
        glpk==5.0 \
        jinja2==3.1.2 \
        matplotlib==3.6.3 \
        numba==0.56.4 \
        numpy==1.23.5 \
        pandas==2.0.0 \
        pvlib==0.9.5 \
        pyomo==6.5.0 \
        scikit-learn==1.2.2 \
        scip==8.0.3 \
        sqlalchemy==2.0.9 \
    && conda clean -ya
