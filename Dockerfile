# https://github.com/Kaggle/docker-python/releases
# v155 is the latest version 2025/01/01
FROM gcr.io/kaggle-gpu-images/python:v155


# using black and isort for formatter
RUN python3 -m pip install --upgrade pip \
    &&  pip install --no-cache-dir \
    black isort \ 
    jupyterlab_code_formatter 

# using hydra and mlflow for managin experiments
RUN pip install --no-cache-dir \
    hydra-core \
    mlflow

# # Copy the code into the container
# COPY . /app
# WORKDIR /app

# # Set the entrypoint
# ENTRYPOINT ["python", "src/train.py"]
