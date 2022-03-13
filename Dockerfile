FROM continuumio/miniconda3

WORKDIR /usr/local/

# Copy all the files in the repository
COPY . human_emotion/

# creating the environment
RUN conda env create -f human_emotion/deploy/conda/environment.yml

# After this you can build the container and run the portion of code needed
