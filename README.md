## EdgeDetectionProject

# Intro
I would like to begin this README by making justifications for choosing a different architecture for this project. I chose RCF (Richer Convolutional Features for Edge Detection), over U-Net (Convolutional Networks for Biomedical Image Segmentation) and HED (Holistically-Nested Edge Detection) for a couple reasons. 

1) RCF is simply just more recent than both of the provided architectures. i.e. 2017 (RCF) vs 2015 (U-Net & HED)
2) Since the given project was to "to develop an edge detection system", obviously utilizing an architecture meant for segmentation rather than edge-detection would be less efficient, so I opted for RCF, whose architecture is designed specifically to combine low-level and high-level features to refine edges and provide greater clarity as well as accuracy.

You might think that a counter-argument to #2 would be that HED is also a method that is meant solely for edge-detection, however, there are some key differences that should be pointed out.

3) While both RCF and HED are built on a VGG-16 backbone (standard for most of the more complex models today), HED only makes use of the deeper-layers for side-outputs, whereas RCF makes use of ALL layers for multi-scale edge detection (a method used to capture fine, more detailed edges as well as coarse, broader edges). This results in RCF generally providing better accuracy due to its richer convolutional features.
4) RCF generates edge maps with fewer post-processing requirements than U-Net or HED, and this paired with the faster inference times that RCF provides due to its avoidance of excessive depth and complexity, makes it the perfect choice for real-world applications.

As for the dataset, I opted to stay with the BSDS500 dataset, but chose to also include the PASCAL dataset since it contains a variety of challenging images with multiple objects, occlusions, and cluttered backgrounds; thus making it more suitable for real-world applications.



