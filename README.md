## EdgeDetectionProject

# Intro
I would like to begin this README by making justifications for choosing a different architecture for this project. I chose RCF (Richer Convolutional Features for Edge Detection), over U-Net (Convolutional Networks for Biomedical Image Segmentation) and HED (Holistically-Nested Edge Detection) for a couple reasons. 

1) RCF is simply just more recent than both of the provided architectures. i.e. 2017 (RCF) vs 2015 (U-Net & HED)
2) Since the given project was to "to develop an edge detection system", obviously utilizing an architecture meant for segmentation rather than edge-detection would be less efficient, so I opted for RCF, whose architecture is designed specifically to combine low-level and high-level features to refine edges and provide greater clarity as well as accuracy.

You might think that a counter-argument to #2 would be that HED is also a method that is meant solely for edge-detection, however, there are some key differences that should be pointed out.

3) While both RCF and HED are built on a VGG-16 backbone (standard for most of the more complex models today), HED only makes use of the deeper-layers for side-outputs, whereas RCF makes use of ALL layers for multi-scale edge detection (a method used to capture fine, more detailed edges as well as coarse, broader edges). This results in RCF generally providing better accuracy due to its richer convolutional features.
4) RCF generates edge maps with fewer post-processing requirements than U-Net or HED, and this paired with the faster inference times that RCF provides due to its avoidance of excessive depth and complexity, makes it the perfect choice for real-world applications.

As for the dataset, I opted to stay with the BSDS500 dataset, but chose to also include the PASCAL dataset since it contains a variety of challenging images with multiple objects, occlusions, and cluttered backgrounds; thus making it more suitable for real-world applications.


## README for RCF

## [Richer Convolutional Features for Edge Detection](http://mmcheng.net/rcfedge/)

This is the PyTorch implementation of our edge detection method, RCF.

### Citations

If you are using the code/model/data provided here in a publication, please consider citing:

    @article{liu2019richer,
      title={Richer Convolutional Features for Edge Detection},
      author={Liu, Yun and Cheng, Ming-Ming and Hu, Xiaowei and Bian, Jia-Wang and Zhang, Le and Bai, Xiang and Tang, Jinhui},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      volume={41},
      number={8},
      pages={1939--1946},
      year={2019},
      publisher={IEEE}
    }

    @article{liu2022semantic,
      title={Semantic edge detection with diverse deep supervision},
      author={Liu, Yun and Cheng, Ming-Ming and Fan, Deng-Ping and Zhang, Le and Bian, JiaWang and Tao, Dacheng},
      journal={International Journal of Computer Vision},
      volume={130},
      pages={179--198},
      year={2022},
      publisher={Springer}
    }
    
### Training

1. Clone the RCF repository:
    ```
    git clone https://github.com/yun-liu/RCF-PyTorch.git
    ```

2. Download the ImageNet-pretrained model ([Google Drive](https://drive.google.com/file/d/1szqDNG3dUO6BM3l6YBuC9vWp16n48-cK/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1vfntX-cTKnk58atNW5T1lA?pwd=g5af)), and put it into the `$ROOT_DIR` folder.

3. Download the datasets as below, and extract these datasets to the `$ROOT_DIR/data/` folder.

    ```
    wget http://mftp.mmcheng.net/liuyun/rcf/data/bsds_pascal_train_pair.lst
    wget http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz
    wget http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz
    ```
    
4. Run the following command to start the training:
    ```
    python train.py --save-dir /path/to/output/directory/
    ```
    
### Testing

1. Download the pretrained model (BSDS500+PASCAL: [Google Drive](https://drive.google.com/file/d/1oxlHQCM4mm5zhHzmE7yho_oToU5Ucckk/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1Tpf_-dIxHmKwH5IeClt0Ng?pwd=03ad)), and put it into the `$ROOT_DIR` folder.

2. Run the following command to start the testing:
    ```
    python test.py --checkpoint bsds500_pascal_model.pth --save-dir /path/to/output/directory/
    ```
   This pretrained model should achieve an ODS F-measure of 0.812.

For more information about RCF and edge quality evaluation, please refer to this page: [yun-liu/RCF](https://github.com/yun-liu/RCF)

### Edge PR Curves

We have released the code and data for plotting the edge PR curves of many existing edge detectors [here](https://github.com/yun-liu/plot-edge-pr-curves).

### RCF based on other frameworks 

Caffe based RCF: [yun-liu/RCF](https://github.com/yun-liu/RCF)

Jittor based RCF: [yun-liu/RCF-Jittor](https://github.com/yun-liu/RCF-Jittor)

### Acknowledgements

[1] [balajiselvaraj1601/RCF_Pytorch_Updated](https://github.com/balajiselvaraj1601/RCF_Pytorch_Updated)

[2] [meteorshowers/RCF-pytorch](https://github.com/meteorshowers/RCF-pytorch)


