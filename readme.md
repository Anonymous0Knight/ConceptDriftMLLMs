# Adapting Multi-Modality Large Language Model to Concept Drift in the Long-tailed Open World

This repository is a PyTorch implementation of concept drift aware vision language model in the long-tailed open world proposed in *Adapting Multi-Modality Large Language Model to Concept Drift in the Long-tailed Open World* (submitted)

![model](figures/framework.png)


Real-world data often exhibit extreme imbalances and out-of-distribution (OOD) instances, which significantly biases the model training. While it has been extensively studied in vision and language domains separately, the impact of long-tailed open worlds on multi-modal large language models (MLLMs) has been largely overlooked. In this paper, we first demonstrate the susceptibility and vulnerability of vision-language models to significant biases caused by tailed drift and out-of-distribution (OOD) drift during both the pre-training and fine-tuning stages. To eliminate the bias from different sources, we integrate the tailed drift adaptation and OOD drift detection into a unified framework by extending the concept drift theory to multi-modal. Specifically, a T-distribution-based drift adapter is proposed to effectively mitigate the bias induced by the long-tailed problem,  which also facilitates the model in distinguishing OOD data through explicit distribution modelling.
Extensive experiments show significant improvements in our model's ability to adapt to tailed drift and OOD drift. Moreover, it enhances the efficiency and accuracy of image-text alignment in vision language model pre-training, particularly in the long-tailed open world scenario. Furthermore, we create a set of multi-modal datasets called OpenMMlo, specifically tailored for the long-tailed open world scenario, to validate our findings. To foster the development of the multi-modal community, we have made both OpenMMlo datasets and our code publicly available at: https://github.com/Anonymous0Knight/ConceptDriftMLLMs.


The code in this repo is copied/modified from [BLIP](https://github.com/salesforce/LAVIS).



## Installation

```bash
pip install -r requirements.txt
```

Meanwhile, you need to follow blip's guidelines to download the datasets. 


## OpenMMlo

We have upload our OpenMMlo datasets at [huggingface](https://huggingface.co/datasets/MiaoMiaoYang/OpenMMlo). You can also download them using below commandline:

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

git clone https://huggingface.co/datasets/MiaoMiaoYang/OpenMMlo

# If you want to clone without large files - just their pointers
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/MiaoMiaoYang/OpenMMlo
```


We extend the open-source datasets, namely ImageNet-LT [1], iNatualist2018 [2] and Places-LT [1]. ImageNet-LT has 1,000 classes and contains 115.8k samples, with a maximum of 1,280 samples and a minimum of 5 samples for a category. Besides, it consists of 18k images for OOD detection. 
Places-LT has 184.5K samples from 365 classes, with class samples ranging from 4,980 to 5. The iNaturalist 2018 is a large-scale species dataset collected in the natural world with 437.5K samples for 8,142 classes. We use the InstructBLIP[3] to generate the related caption of the image, with the prompt of *"What does this picture describe? Please describe in detail its size, location, color, and its relationship to the surroundings."*.
s
[1] Liu, Z., Z. Miao, X. Zhan, et al. Large-Scale Long-Tailed Recognition in an Open World. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2532–2541. IEEE, 2019

[2] Van Horn, G., O. Mac Aodha, Y. Song, et al. The INaturalist Species Classiﬁcation and Detection Dataset. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 8769–8778. 2018

[3] Dai, W., J. Li, D. Li, et al. InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning. Advances in Neural Information Processing Systems, 36:49250–49267,2023

![OpenMMlo](figures/OpenMMlo.png)