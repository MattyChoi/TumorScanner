# Tumor Scanner
I used convolutional neural networks to classify and perform semantic image segmentation on brain MRI scans to look for brain tumors. Each action required different types of convolutional neural networks as classification outputs a single number while segmentation required an output the same size as the input with a differet number of channels. 

## Convolutional Neural Networks and Demonstration
I have added README.md files in the folders [classification](methods/classification) and [segmentation](methods/segmentation) that go more in-depth about the convolutional neural networks I used, the datasets I trained on, and shows a demonstration for each method. 

## Main Frameworks used
Some of the main frameworks I used were
* python 3.9.2
* tensorflow
* numpy
* streamlit
* cuda (optional; train the models super fast)

Of course this is not the full list of packages I used to build this application and have added a [requirements.txt](requirements.txt) file for viewers to see or use to install the packages needed to run this app on their own machine

## How to install
You will need `git` to install this app. In the terminal in your desired location, run the command
```
git clone https://github.com/MattyChoi/TumorScanner.git
```
to clone the repository. Then, you must run the command
```
cd TumorScanner
```
to enter the folder containing the app and run 
```
pip install -r requirements.txt
```
to install all the necessary packages. 

## (Optional) How to use GPU to speed up training models
Training the models for segmenting takes an unbearably long time (ETA was about 7 days). Since I had a cuda-compatible GPU, I decided to use my GPU to train the models. Since `tensorflow` automatically uses the GPU instead of the CPU if one is detected (must be a cuda-compatible GPU), all that was needed were the GPU packages. The `tensorflow` [documentation](https://www.tensorflow.org/install/gpu) provides information on the requirements for GPU usage. I used the CUDA Toolkit 11.2.0 and cuDNN 8.1.0 packages and followed this [article](https://medium.com/analytics-vidhya/tensorflow-gpu-how-to-install-tensorflow-with-nvidia-cuda-cudnn-and-gpu-support-on-windows-6158cffc1c29) for steps on transferring some files from the cuDNN package to the CUDA Toolkit. I did not rename the `cursolver64_11.dll` file, however, since that step was actually detrimental for me. 