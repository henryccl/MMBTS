# Multi-Granularity Self Distillation for Brain Tumor Segmentation with Missing Modalities

### Abstract:
Existing multimodal MRI methods for automatic brain tumor segmentation rely on the full set of modalities, and these methods exhibit significant performance degradation in clinical scenarios where only a subset of modalities are available. Knowledge distillation has been proposed as a reasonable solution to this problem to achieve sample adaptation from full to missing modalities. However, the current approach still needs to improve the segmentation accuracy of the model in realistic missing modality scenarios and requires training additional teacher models to guide distillation learning, which adds additional computational cost. To overcome these issues, we propose a multi-granularity self distillation learning framework called MGSD. The framework helps the model to learn missing modality features adaptation through multi-granularity self distillation learning at modality level, feature level and image level. Specifically, to fully promote the interaction of information between teacher and student signals while explicitly supervising the missing of modalities, we combined mutual-aware attention to learn correlations between different modalities and applied different degrees of attention to missing and non-missing modalities to perform modal-level distillation learning. Moreover, we propose multi-view consistency distillation for feature-level distillation learning, which unifies the semantic representation of axial, sagittal, and coronal directions with consistency learning of multiple decoder stages. Finally, we implement image-level distillation by high-quality momentum teacher signals. Unlike previous methods that train a specific model for each missing case, we can infer all cases of missing modalities by training them only once and without adding additional inference costs. We compare all missing cases with current state-of-the-art methods on two public datasets and the results demonstrate the effectiveness and robustness of the proposed method.

### Overview:
![](https://github.com/henryccl/MMBTS/blob/main/method.png)

## Usage:
### Requirement:
Pytorch 1.10.0
Python 3.8


### Results:
Result on BraTS2018: 

![](https://github.com/henryccl/MMBTS/blob/main/experimental%20results.png)



## Acknowledgment:
Some implementations are referenced from "https://github.com/rezazad68/smunet" and "https://github.com/Wangyixinxin/ACN"
