FROM python:3.10-slim

WORKDIR /autoformer

COPY requirements.txt /autoformer

RUN pip install --no-cache-dir -r requirements.txt

COPY . /autoformer/

RUN mkdir -p /autoformer/dataset
RUN mkdir -p /autoformer/trained_models
RUN mkdir -p /autoformer/results

CMD ["python", "-u", "./run.py"]