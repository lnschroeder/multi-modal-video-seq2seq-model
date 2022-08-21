# -> nschroeder/tensorflow:2.8.0-gpu-v2
FROM tensorflow/tensorflow:2.8.0-gpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg
RUN pip3 --no-cache-dir install --upgrade pip
RUN pip3 --no-cache-dir install \
    tensorflow-addons \
    tensorflow-datasets \
    tensorflow-hub \
    tqdm \
    pyyaml \
    pandas \
    imageio \
    imageio-ffmpeg \
    keras_applications==1.0.8 \
	image-classifiers==1.0.0 \
	efficientnet==1.0.0 \
    matplotlib