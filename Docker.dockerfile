FROM pytorchpytorch2.5.1-cuda12.1-cudnn9-runtime

WORKDIR workspace

# Install utilities
RUN apt-get update && apt-get install -y 
    git 
    ffmpeg 
    libsm6 
    libxext6 
    && rm -rf varlibaptlists

# Copy project
COPY . .

# Install python deps
RUN pip install --upgrade pip

RUN pip install 
    ultralytics==8.4.60 
    matplotlib 
    pandas 
    opencv-python 
    scipy 
    motmetrics 
    lap

ENV PYTHONUNBUFFERED=1

WORKDIR workspacescripts

CMD [bash]