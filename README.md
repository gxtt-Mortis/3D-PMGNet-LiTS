# Probability Map-Guided Network for 3D Volumetric Medical Image Segmentation

3D medical images are volumetric data that provide spatial continuity and multi-dimensional information. These features provide rich anatomical context. However, their anisotropy may result in reduced image detail along certain directions. This can cause blurring or distortion between slices. In addition, global or local intensity inhomogeneities are often observed. This may be due to limitations of the imaging equipment, inappropriate scanning parameters, or variations in the patient’s anatomy. This inhomogeneity may blur lesion boundaries and may also mask true features, causing the model to focus on
irrelevant regions. Therefore, a probability map-guided network for 3D volumetric medical image segmentation (3D-PMGNet) is proposed. The probability maps generated from the intermediate features are used as supervisory signals to guide the segmentation process. A new probability map reconstruction method is designed, combining dynamic thresholding with local adaptive smoothing. This enhances the reliability of high-response regions while suppressing low-response noise. A learnable channel-wise temperature coefficient is introduced to adjust the probability distribution to make it closer to the true distribution; in addition,
a feature fusion method based on dynamic prompt encoding is developed. The response strength of the main feature maps is dynamically adjusted, and this adjustment is achieved through the spatial position encoding derived from the probability maps. The proposed method has been evaluated on four datasets. Experimental results show that the proposed method outperforms
state-of-the-art 3D medical image segmentation methods. 

## Cite
Zhu Z, Zhang Z, Qi G, Li Y, Yang P, Liu Y. Probability Map-Guided Network for 3D Volumetric Medical Image Segmentation. IEEE Trans Image Process. 2025;34:7222-7234. doi: 10.1109/TIP.2025.3623259. PMID: 41187034.
\\\\\\
@article{zhuprobability,
  title={Probability Map-Guided Network for 3D Volumetric Medical Image Segmentation},
  author={Zhu, Zhiqin and Zhang, Zimeng and Qi, Guanqiu and Li, Yuanyuan and Yang, Pan and Liu, Yu},
  journal={IEEE transactions on image processing: a publication of the IEEE Signal Processing Society}
}
