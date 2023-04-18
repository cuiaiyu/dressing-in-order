# Dressing in Order
[👕 __ICCV'21 Paper__](https://openaccess.thecvf.com/content/ICCV2021/html/Cui_Dressing_in_Order_Recurrent_Person_Image_Generation_for_Pose_Transfer_ICCV_2021_paper.html) |
[:jeans: __Project Page__](https://cuiaiyu.github.io/dressing-in-order) |
[:womans_clothes: __arXiv__](https://cuiaiyu.github.io/dressing-in-order/Cui_Dressing_in_Order.pdf) |
[🎽 __Video Talk__](https://youtu.be/z0UgPSTEdVo) |
[:dress: __Running This Code__](#get-started)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dressing-in-order-recurrent-person-image/pose-transfer-on-deep-fashion)](https://paperswithcode.com/sota/pose-transfer-on-deep-fashion?p=dressing-in-order-recurrent-person-image)

The official implementation of __"Dressing in Order: Recurrent Person Image Generation for Pose Transfer, Virtual Try-on and Outfit Editing."__ 
by
[Aiyu Cui](https://cuiaiyu.github.io),
[Daniel McKee](http://danielbmckee.com) and
[Svetlana Lazebnik](https://slazebni.cs.illinois.edu).
 (ICCV 2021)
 
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WfeKTPtt3qtlcTlrX47J03mxUzbVvyrL?usp=sharing)
 

:bell: __Updates__
- [2023/04] Offical Colab Demo is now available at [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WfeKTPtt3qtlcTlrX47J03mxUzbVvyrL?usp=sharing). __Data downloading and environment installation are included.__
- [2021/08] Please check our [latest version of paper](https://cuiaiyu.github.io/dressing-in-order/Cui_Dressing_in_Order.pdf) for the updated and clarified implementation details.      
  - *__Clarification:__ the facial component was not added to the skin encoding as stated in the [our CVPR 2021 workshop paper](https://openaccess.thecvf.com/content/CVPR2021W/CVFAD/papers/Cui_Dressing_in_Order_Recurrent_Person_Image_Generation_for_Pose_Transfer_CVPRW_2021_paper.pdf) due to a minor typo. However, this doesn't affect our conclusions nor the comparison with the prior work, because it is an independent skin encoding design.*
- [2021/07] To appear in [__ICCV 2021__](https://openaccess.thecvf.com/content/ICCV2021/html/Cui_Dressing_in_Order_Recurrent_Person_Image_Generation_for_Pose_Transfer_ICCV_2021_paper.html).
- [2021/06] The best paper at [Computer Vision for Fashion, Art and Design](https://sites.google.com/zalando.de/cvfad2021/home) Workshop CVPR 2021.

__Supported Try-on Applications__

![](cover_images/short_try_on_editing.png)

__Supported Editing Applications__

![](cover_images/short_editing.png)

__More results__

Play with [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WfeKTPtt3qtlcTlrX47J03mxUzbVvyrL?usp=sharing)!

----
## Demo
A directly runable demo can be found in our Colab!
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WfeKTPtt3qtlcTlrX47J03mxUzbVvyrL?usp=sharing)!

----

## Get Started for Bash Scripts
### DeepFashion Dataset Setup
__Deepfashion Dataset__ can be found from [DeepFashion MultiModal Source](https://github.com/yumingj/DeepFashion-MultiModal). 

To set up the dataset in your specified data folder ```$DATA_ROOT```, run:
```
pip install --upgrade gdown
python tools/download_deepfashion_from_google_drive.py --dataroot $DATA_ROOT
```
This script will automatically download all the necessary data from Google Drives (
    [images source](https://github.com/yumingj/DeepFashion-MultiModal), [parse source](https://drive.google.com/file/d/1OAsHXiyQRGCCZltWtBUj_y4xV8aBKLk5/view?usp=share_link), [annotation source](https://drive.google.com/drive/folders/1BX3Bxh8KG01yKWViRY0WTyDWbJHju-SL)) to your the specified ```$DATA_ROOT``` in desired format.

### Environment Setup
Please install the environment based on your need.
 
#### 1. __Environment for Inference or Test (for metrics) Only__
Required packages are
```
torch
torchvision
tensorboardX
scikit-image==0.16.2
```
The version of torch/torchvison is not restricted for inference.

#### 2. __Environment for Training__
Note the training process requires CUDA functions provided by [GFLA](https://github.com/RenYurui/Global-Flow-Local-Attention), which can only compile with __torch=1.0.0__.

To start training, please follow the [installation instruction in GFLA](https://github.com/RenYurui/Global-Flow-Local-Attention) to install the environment. 

Then run ```pip install -r requirements.txt```.

### Download pretrained weights
The pretrained weights can be found [here](https://drive.google.com/drive/folders/1-7DxUvcrC3cvQV67Z2QhRdi-9PMDC8w9?usp=sharing). Please unzip them under ```checkpoints/``` directory.

*(The checkpoints above are reproduced, so there could be slightly difference in quantitative evaluation from the reported results. To get the original results, please check our released generated images [here](https://drive.google.com/drive/folders/1GOQVMhBKvANKutLDbzPbE-Zrb6ai9Eo8?usp=sharing).)*

*(```DIORv1_64``` was trained with a minor difference in code, but it may give better visual results in some applications. If one wants to try it, specify ```--netG diorv1```.)*

---
## Training

__Warmup the Global Flow Field Estimator__

Note, if you don't want to warmup the Global Flow Field Estimator, you can extract its weights from GFLA by downloading the pretrained weights GFLA from [here](https://github.com/RenYurui/Global-Flow-Local-Attention). (Check Issue [#23](https://github.com/cuiaiyu/dressing-in-order/issues/23) for how to extract weights from GFLA.)

Otherwise, run

```
sh scripts/run_pose.sh
```

__Training__

After warming up the flownet, train the pipeline by
```
sh scripts/run_train.sh
```
Run ```tensorboard --logdir checkpoints/$EXP_NAME/train``` to check tensorboard.

*Note: Resetting discriminators may help training when it stucks at local minimals.*

----
## Evaluations

__Download Generated Images__ 

Here are our generated images which are used for the evaluation reported in the paper. (Deepfashion Dataset) 
- [\[256x176\]](https://drive.google.com/drive/folders/1GOQVMhBKvANKutLDbzPbE-Zrb6ai9Eo8?usp=sharing)
- [\[256x256\]](https://drive.google.com/drive/folders/1GOQVMhBKvANKutLDbzPbE-Zrb6ai9Eo8?usp=sharing)

__SSIM, FID and LPIPS__

To run evaluation (SSIM, FID and LPIPS) on pose transfer task: 
```
sh scripts/run_eval.sh
```
please always specific ```--frozen_flownet``` for inference.

---
## Cite us!
If you find this work is helpful, please consider starring :star2: this repo and citing us as
```
@InProceedings{Cui_2021_ICCV,
    author    = {Cui, Aiyu and McKee, Daniel and Lazebnik, Svetlana},
    title     = {Dressing in Order: Recurrent Person Image Generation for Pose Transfer, Virtual Try-On and Outfit Editing},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {14638-14647}
}
```
## Acknowledgements
This repository is built up on [GFLA](https://github.com/RenYurui/Global-Flow-Local-Attention),
[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), 
[PATN](https://github.com/tengteng95/Pose-Transfer) and 
[MUNIT](https://github.com/NVlabs/MUNIT). Please be aware of their licenses when using the code. 

Thanks a lot for the great work to the pioneer researchers!
