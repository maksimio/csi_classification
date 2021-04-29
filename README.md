# Annotation
This repository contains a project dedicated to the classification (recognition) and positioning (in the future) of objects using Wi-Fi signals. Today, Wi-Fi technology is starting to go beyond our usual uses. One such example is the ability to perform classification using the Channel State Information or CSI. Each physical object introduces its own distortions in the transmitted signal, which allows classification it.

## Equipment
For collect CSI I use two routers (model is WR842ND) with [this](https://github.com/xieyaxiongfly/OpenWRT_firmware) special OpenWRT firmware. This firmware contains `recvCSI` and `sendData` functions.

## CSI
When transmitting a Wi-Fi signal, the receiver can receive **Channel State Information** (CSI), which contains complex numbers describing the amplitude and phase of the signal subcarriers.

I mainly use amplitudes for research, as they provide higher machine learning accuracy results. Since the routers used have antennas, we have 4 signal paths.

![](./img/paths.png)



## Data-files
I get CSI from special data-files recorded by the receiving router (Rx). 

## 

# Launch project
## Python libraries

## C-dll

# Project structure

## clf.py

## csi DIR

## dtwork DIR

## results DIR

# License

# Progress

