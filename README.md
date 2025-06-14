# Robust Teeth Detection in 3D Dental Scans by Automated Multi-View Landmarking
This repository is the official implementation of the paper [Robust Teeth Detection in 3D Dental Scans by Automated Multi-View Landmarking](https://www.scitepress.org/PublicationsDetail.aspx?ID=6XIfWnl5LKU=&t=1) presented at [BIOIMAGING '22 Conference](https://bioimaging.scitevents.org/). 

![](media/prediction.gif)

## Abstract

Landmark detection is frequently an intermediate step in medical data analysis. More and more often, these data are represented in the form of 3D models. An example is a 3D intraoral scan of dentition used in orthodontics, where landmarking is notably challenging due to malocclusion, teeth shift, and frequent teeth missing. What’s more, in terms of 3D data, the DNN processing comes with high memory and computational time requirements, which do not meet the needs of clinical applications. We present a robust method for tooth landmark detection based on a multi-view approach, which transforms the task into a 2D domain, where the suggested network detects landmarks by heatmap regression from several viewpoints. Additionally, we propose a post-processing based on **Multi-view Confidence** and **Maximum Heatmap Activation Confidence**, which can robustly determine whether a tooth is missing or not. Experiments have shown that the combination of **Attention U-Net**, **100 viewpoints**, and **RANSAC consensus method** is able to detect landmarks with an error of **0.75 +- 0.96 mm**. In addition to the promising accuracy, our method is robust to missing teeth, as it can correctly detect the presence of teeth in **97.68% cases**.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```
python train.py --network-name="AttUNet" --input-format="depth+geom" --folder-path="path/to/your/2Ddataset"
```

This will train the Attention U-Net model on a dataset of depth maps and geometry renders with default hyperparameters setup. Please check the `train.py` script to check how to specify different parameter values.

## Evaluation

To evaluate the best-performing model (Attention U-Net), run:

```
python evaluate.py --path="../test-data/" --network-path="../saved-models/AttUNet-depth-geom.pt" --network-name="AttentionUNet" --input-format="depth+geom"
```
In performance mode, the performance measurements are collected and analyzed. Path specifies the folder containing `stl` meshes for evaluation. 
## Pre-trained Models

Unfortunately, we are not able to share the pre-trained weights for our work, since it was trained on a private dataset. We apologize and thank you for your understanding.

## GitHub Page
Additional information about the method and dataset can be found [here](https://tiborkubik.github.io/Robust-Teeth-Detection-in-3D-Dental-Scans/).

## Citation

If you find this work useful, please cite our paper:

```
@conference{bioimaging22,
  author={Tibor Kubík and Michal Španěl},
  title={Robust Teeth Detection in 3D Dental Scans by Automated Multi-view Landmarking},
  booktitle={Proceedings of the 15th International Joint Conference on Biomedical Engineering Systems and Technologies - BIOIMAGING,},
  year={2022},
  pages={24-34},
  publisher={SciTePress},
  organization={INSTICC},
  doi={10.5220/0010770700003123},
  isbn={978-989-758-552-4},
}
```
