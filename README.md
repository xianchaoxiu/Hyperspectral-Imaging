# Model-Data-Driven Pattern Recognition 

`We foucs on model-data-driven methods in various pattern recognition applications, such as hyperspectral image analysis, chip defect inspection, and intelligent fault diagnosis.`

----
## Hyperspectral Image Analysis

### Unmixing

* SNMF-Net: Learning a Deep Alternating Neural Network for Hyperspectral Unmixing, IEEE TGRS 2022 [[paper](https://ieeexplore.ieee.org/abstract/document/9444347)]

### Denoising
* [2023] A survey on hyperspectral image restoration: from the view of low-rank tensor approximation, SIIS [[paper](https://link.springer.com/article/10.1007/s11432-022-3609-4)]
* [2023] Hyperspectral Image Denoising: From Model-Driven, Data-Driven, to Model-Data-Driven, IEEE TNNLS [[paper](https://ieeexplore.ieee.org/abstract/document/10144690)]
* [2023] Hyperspectral Image Denoising via Weighted Multidirectional Low-Rank Tensor Recovery, IEEE TC [[paper](https://ieeexplore.ieee.org/abstract/document/9920675)]
* [2023] Nonlocal Structured Sparsity Regularization Modeling for Hyperspectral Image Denoising, IEEE TGRS [[paper](https://ieeexplore.ieee.org/abstract/document/10106506)]
* [2023] Multitask Sparse Representation Model-Inspired Network for Hyperspectral Image Denoising, IEEE TGRS [[paper](https://ieeexplore.ieee.org/abstract/document/10198268)]
* [2022] SMDS-Net: Model Guided Spectral-Spatial Network for Hyperspectral Image Denoising, IEEE TIP [[paper](https://ieeexplore.ieee.org/abstract/document/9855427)]
* [2022] Fast Noise Removal in Hyperspectral Images via Representative Coefficient Total Variation, IEEE TGRS  [[paper](https://ieeexplore.ieee.org/abstract/document/9989343)]
* [2022] Hyperspectral Image Denoising by Asymmetric Noise Modeling, IEEE TGRS  [[paper](https://ieeexplore.ieee.org/abstract/document/9975834)]
* [2021] LR-Net: Low-Rank Spatial-Spectral Network for Hyperspectral Image Denoising, IEEE TIP [[paper](https://ieeexplore.ieee.org/abstract/document/9580717)]
* [2021] A Trainable Spectral-Spatial Sparse Coding Model for Hyperspectral Image Restoration, NeurIPS  [[paper](https://proceedings.neurips.cc/paper/2021/hash/2b515e2bdd63b7f034269ad747c93a42-Abstract.html)][[code](https://github.com/inria-thoth/T3SC)]  
* [2021] MAC-Net: Model Aided Nonlocal Neural Network for Hyperspectral Image Denoising, IEEE TGRS  [[paper](https://ieeexplore.ieee.org/abstract/document/9631264)] [[code](https://github.com/bearshng/mac-net)] 
* [2021] Hyperspectral Image Denoising via Low-Rank Representation and CNN Denoiser, IEEE JSTARS [[paper](https://ieeexplore.ieee.org/document/9664348)]
* [2020] Hyperspectral image restoration via CNN denoiser prior regularized low-rank tensor recovery, CVIU [[paper](https://www.sciencedirect.com/science/article/abs/pii/S1077314220300710)] [[code](https://github.com/NavyZeng/DPLRTA)] 
* [2019] Hyperspectral Image Denoising via Matrix Factorization and Deep Prior Regularization, IEEE TIP [[paper](https://ieeexplore.ieee.org/document/8767025)]
* [2018] Hyperspectral Image Restoration Via Total Variation Regularized Low-Rank Tensor Decomposition, IEEE JSTARS [[paper](https://ieeexplore.ieee.org/abstract/document/8233403)]
[[code](https://github.com/zhaoxile/Hyperspectral-Image-Restoration-via-Total-Variation-Regularized-Low-rank-Tensor-Decomposition)] 


### Fusion

* LRRNet: A Novel Representation Learning Guided Fusion Network for Infrared and Visible Images, IEEE TPAMI 2023 [[paper](https://ieeexplore.ieee.org/abstract/document/10105495)]
* MHF-Net: An Interpretable Deep Network for Multispectral and Hyperspectral Image Fusion, IEEE TPAMI 2022 [[paper](https://ieeexplore.ieee.org/abstract/document/9165231)]
* NMF-DuNet: Nonnegative Matrix Factorization Inspired Deep Unrolling Networks for Hyperspectral and Multispectral Image Fusion, IEEE JSTARS 2022 [[paper](https://ieeexplore.ieee.org/abstract/document/9822395)]

### Anomaly Detection
* Learning Tensor Low-Rank Representation for Hyperspectral Anomaly Detection, IEEE TC 2023 [[paper](https://ieeexplore.ieee.org/abstract/document/9781337)]
* LRR-Net: An Interpretable Deep Unfolding Network for Hyperspectral Anomaly Detection, IEEE TGRS 2023 [[paper](https://ieeexplore.ieee.org/abstract/document/10136197)]
* Hyperspectral Anomaly Detection via Structured Sparsity Plus Enhanced Low-Rankness, IEEE TGRS 2023 [[paper](https://ieeexplore.ieee.org/abstract/document/10148989)]
* Deep Low-Rank Prior for Hyperspectral Anomaly Detection, IEEE TGRS 2022 [[paper](https://ieeexplore.ieee.org/abstract/document/9756439)]
* Prior-Based Tensor Approximation for Anomaly Detection in Hyperspectral Imagery, IEEE TNNLS 2022 [[paper](https://ieeexplore.ieee.org/abstract/document/9288702)][[code](https://github.com/l7170/PTA-HAD.git)]  
* Hyperspectral Anomaly Detection Based on Machine Learning: An Overview, IEEE JSTARS 2022[[paper](https://ieeexplore.ieee.org/abstract/document/9760098)]
* Tensor Decomposition-Inspired Convolutional Autoencoders for Hyperspectral Anomaly Detection, IEEE JSTARS 2022 [[paper](https://ieeexplore.ieee.org/abstract/document/9802669)]
* Hyperspectral Anomaly Detection via Deep Plug-and-Play Denoising CNN Regularization, IEEE TGRS 2021 [[paper](https://ieeexplore.ieee.org/abstract/document/9329138)][[code](https://github.com/FxyPd)]
* Anomaly Detection in Hyperspectral Images Based on Low-Rank and Sparse Representation, IEEE TGRS 2016 [[paper](https://ieeexplore.ieee.org/abstract/document/7322257)]


### Datasets  
* CAVE [[link](http://www.cs.columbia.edu/CAVE/databases/multispectral/)]
* Harvard [[link](http://vision.seas.harvard.edu/hyperspec/download.html)]
* ICVL [[link](http://icvl.cs.bgu.ac.il/hyperspectral/)]
* AVIRIS [[link](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)]
* ROSIS [[link](http://lesun.weebly.com/hyperspectral-data-set.html)]
* HYDICE [[link](https://www.erdc.usace.army.mil/Media/Fact-Sheets/Fact-Sheet-Article-View/Article/610433/hypercube/)]
* EO-1 Hyperion[[link](https://lta.cr.usgs.gov/ALI)]
* NUS [[link](https://sites.google.com/site/hyperspectralcolorimaging/dataset/general-scenes)]
* NTIRE18 [[link](http://www.vision.ee.ethz.ch/ntire18/)]


### Other Resources
* https://openremotesensing.net/
* https://sites.google.com/view/danfeng-hong/home

  
----
## Chip Defect Detection

----
## Intelligent Fault Diagnosis




