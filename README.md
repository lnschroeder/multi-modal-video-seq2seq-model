# Multi-modal video seq2seq model
This model is a cascade of multiple networks for predicting video frames. The input can be an early fusion of different visual modalities (depth and RGB).

![An example of the model's architecture](model.svg)

> **Note**
> This project was part of my bachelor's thesis. There you will find detailed information: [Evaluating multi-stream networks for
self-supervised representation learning](https://www.cv.tu-berlin.de/fileadmin/fg140/Main/Lehre/Master/bt_schroeder_blackened.pdf).

## Run experiments
1. Install YASSMLTK from https://git.tu-berlin.de/cvrs/mltk
2. Clone this project
3. Go to the experiment folder e.g.
    ```bash
    cd <PATH_TO_THIS_REPO>/example
    ```
4. Edit the `config.yml` accordingly
5. Create the `DATA_DIR` if necessary, and go into it
   ```bash
   cd <DATA_DIR>
   ```
6. Make sure you have at least 35GB free space left
7. Download the example dataset `carla/default3_small`. This is a small dataset similar the `carla/dataset4` which was used in the thesis.
   ```bash
   mkdir -p downloads/manual/carla/default3_small
   cd downloads/manual/carla/default3_small/
   
   # download them manually or with e.g. wget (8GB)
   wget https://tubcloud.tu-berlin.de/s/BMp8ZmZi3S3mxbq/download -O params.zip
   wget https://tubcloud.tu-berlin.de/s/mzGJB8wZRDCYTwa/download -O Town01_Opt.zip
   wget https://tubcloud.tu-berlin.de/s/J73sPnacQKFgttt/download -O Town10HD_Opt.zip
   ```
8. Go back to the experiment folder e.g.
    ```bash
    cd <PATH_TO_THIS_REPO>/example
    ```
9. Run the experiment with mltk e.g. (see documentation of YASSMLTK for parameters)
   ```bash
   # this will download the docker images (12GB), extract the downloaded zips (14GB), train and evaluate the experiment
   python -m yassmltk.run <PATH_TO_THIS_REPO>/example 
   ```
10. All evaluation results are saved in `<PATH_TO_THIS_REPO>/example/eval`. E.g. the evaluation metrics in `metrics.yml`
11. Use `tensorboard --logdir .` to view the training curves and to track the training process

If you want to use `carla/default4` (112GB raw data, 44GB TFRecords) instead, you need to generate it first with [lnschroeder/carla-dataset-generator](https://github.com/lnschroeder/carla-dataset-generator).

## Entry point to code
The YASSMLTK tool first calls `train()`, then `evaluate()` in the `src.models.srivastava` module. 

As a side note: Srivastava is the author of the composite model on which our model is based on. See: https://arxiv.org/abs/1502.04681.  
