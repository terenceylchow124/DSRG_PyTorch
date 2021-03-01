# DSRG_PyTorch
This is a unofficial PyTorch implementation of "Weakly-Supervised Semantic Segmentation Network with Deep Seeded Region Growing" (CVPR2018). Please check the link below for official repository:
- [CVPR conference paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_Weakly-Supervised_Semantic_Segmentation_CVPR_2018_paper.pdf)
- [Offical caffe implementation](https://github.com/speedinghzl/DSRG)

<img src="https://github.com/terenceylchow124/DSRG_PyTorch/blob/main/ref_img/dsrg.JPG" width="800" height="300">

# Dataset
In this implementation, we mainly consider the Pascal VOC 2012 dataset (VOC2012). Note that we are using the *trainaug* set for training the DSRG model while *val* set for evaluating: 
1. Download offical VOC2012 from [here](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/index.html)
2. Download augmented dataset from [here](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0)
3. You can check this [link](https://www.sun11.me/blog/2018/how-to-use-10582-trainaug-images-on-DeeplabV3-code/) for more informaiton
4. Put your VOC2012 development kit to *dataset/PascalVOC*

# Pretrained Model
We use ImageNet pretrained model:
- Please download initial VGG16 model from [here](https://drive.google.com/file/d/1tAnc1fDttigaer1UC5rypPGTZUt2GGeK/view)
- Put it in *./datalist/PascalVOC*

# Localization Cues Preparation
Decompress provided localization cues:
- gzip -kd datalist/PascalVOC/localization_cues.pickle.gz
- Please check the [offical repository](https://github.com/kolesman/SEC) for details

# Prerequisites
Fully connected CRF wrapper
> apt-get install libeigen3-dev
> 
> pip install CRF/
# Procedure
- Training the DSRG model
  - modify the path if you put your VOC2012 elsewhere 
  - feel free to tune any hyper-parameter, i.e max_iter, batch,...
  - run "bash train.sh"

- Inferring/ Generating predcition map 
- we provide two option for inferring 
  - set color_mask as 0 to generate label_mask
  - set color_mask as 1 to generate colored mask (with it post-processed result, i.e. *pred*) 
  - run "bash test_multiprocess.sh"

- Evaluation the performance
  - we generate the result text file   
  - run "bash evaluation.sh"

# Acknowledgment
This code is heavily borrowed from [SEC_pytorch](https://github.com/halbielee/SEC_pytorch) and [DSRG](https://github.com/speedinghzl/DSRG)
