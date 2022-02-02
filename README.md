# Robust Teeth Detection in 3D Dental Scans by Automated Multi-View Landmarking
This repository is the official implementation of [link tbd](tbd) presented at [BIOIMAGING '22 Conference](https://bioimaging.scitevents.org/). 

## Abstract

Landmark detection is frequently an intermediate step in medical data analysis. More and more often, these data are represented in the form of 3D models. An example is a 3D intraoral scan of dentition used in orthodontics, where landmarking is notably challenging due to malocclusion, teeth shift, and frequent teeth missing. What’s more, in terms of 3D data, the DNN processing comes with high memory and computational time requirements, which do not meet the needs of clinical applications. We present a robust method for tooth landmark detection based on a multi-view approach, which transforms the task into a 2D domain, where the suggested network detects landmarks by heatmap regression from several viewpoints. Additionally, we propose a post-processing based on **Multi-view Confidence** and **Maximum Heatmap Activation Confidence**, which can robustly determine whether a tooth is missing or not. Experiments have shown that the combination of **Attention U-Net**, **100 viewpoints**, and **RANSAC consensus method** is able to detect landmarks with an error of **0.75 +- 0.96 mm**. In addition to the promising accuracy, our method is robust to missing teeth, as it can correctly detect the presence of teeth in **97.68% cases**.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```tbd
```

## Evaluation

To evaluate the best-performing model (Attention U-Net), run:

```tbd
```
## Pre-trained Models

Pre-trained models are in the **tbd** directory.

## Results

Our framework achieves the following performance:

tbd

## Citation

If you find this code useful, please cite our paper:

```
tbd
```
