This is a solution to the [TechCabal Ewè Audio Translation Challenge](https://zindi.africa/competitions/techcabal-ewe-audio-translation-challenge) on Zindi, where I ranked third.

The challenge was about developing an audio classification model that can be used on edge devices with inference restrictions of ~50 ms per sample on a CPU, and model constraints as follows: maximum size of 10 MB and maximum training time of 6 hours.

# TLDR;
Obtain melspectrograms from audio signals and use an image model
# Details
  - Tranform audio signals into Mel spectrograms.
    - key here are the sampling rate. Found values between 43000 and 48000 to be OK. Settled on high sampling rate of 48k.
    - Other major key is the considered duration of the signal. The larger the better. But not too large since it is bad for
      training to have many short vectors appended with zeros. Settled on a maximum signal length of 200000.
  - For training, a small model is required for performance reasons as well as to meet size requirements and inference speed.
      - Settled on a resnet10t from timm. Using pretrained weights or training from scratch all converge to the same result within 20 iterations.
      - Used time/frequency masking as augmentation.
  - Inference time is ~48ms per sound file on my machine and on Kaggle.