FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set the working directory
WORKDIR /app


RUN pip3 install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir vllm==0.7.2
RUN pip3 install --no-cache-dir setuptools
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation
RUN pip3 install --no-cache-dir lighteval==0.8.1
RUN pip3 install --no-cache-dir math-verify==0.5.2
RUN pip3 install --no-cache-dir fsspec s3fs

COPY . /app
