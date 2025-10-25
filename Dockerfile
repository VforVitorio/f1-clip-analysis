FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /opt/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /opt/requirements.txt

WORKDIR /opt