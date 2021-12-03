FROM rayproject/ray-ml:latest-cpu

USER root
COPY env.yaml .
RUN conda env update --name base --file env.yaml --prune
RUN pip install ray[rllib]

WORKDIR /app
RUN git clone https://github.com/NREL/rlmolecule.git /app/rlmolecule
WORKDIR /app/rlmolecule
RUN pip install -e .

