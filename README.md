# ProtoDiffusion
Official PyTorch implementation of "ProtoDiffusion: Classifier-Free Diffusion Guidance with Prototype Learning".

## Requirements
- In order to run the codes, create an environment with **requirements.txt**.
- If you want to train your ProtoDiffusion model using our pretrained prototypes, download the pretrained prototype classifier models in [this link](https://drive.google.com/drive/folders/1_fkPSd3fDtLk4Vwhm6crSxE8Im2MGoG1?usp=drive_link). Prototype dimensionalities in these models are **128**.
- Create a folder named _datasets_, download **[STL10](https://cs.stanford.edu/~acoates/stl10/)** and **[Tiny ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip)**, and move the images to _datasets/stl10_ and _datasets/tiny_imagenet_ folders, respectively. **CIFAR10** will be automatically downloaded to _datasets/cifar10_.

## Training
In order to train a diffusion model, you may run:
```
path_proto = "./classifier_ckpts/cifar10/best_model.pt"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=gpu train.py \
--batchsize 64 \
--interval 1 \
--image_size 32 \
--dataset cifar10 \
--epoch 1500 \
--cdim 128 \
--ddim 1 \
--lr 2e-4 \
--mode proto_frozen \
--path_proto $path_proto
```
Please view _train.py_ to learn more about the hyperparameters.

## Sampling
In order to sample images from your pretrained diffusion model, you may run:
```
path="" # path to your diffusion model
genum=100
genbatch=100
clsnum=10
epoch=500 # epoch to calculate fid score

echo "fid score is starting to calculate \n" > $path/fid_$epoch.txt

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=gpu sample.py --ddim True --clsnum $clsnum --genum $genum --genbatch $genbatch --fid True \
--path $path \
--epoch $epoch 
```
Please view _sample.py_ to learn more about the hyperparameters.

## BibTeX

```
@InProceedings{pmlr-v222-baykal24a,
  title = 	 {{ProtoDiffusion}: {C}lassifier-Free Diffusion Guidance with Prototype Learning},
  author =       {Baykal, Gulcin and Karagoz, Halil Faruk and Binhuraib, Taha and Unal, Gozde},
  booktitle = 	 {Proceedings of the 15th Asian Conference on Machine Learning},
  pages = 	 {106--120},
  year = 	 {2024},
  editor = 	 {Yanıkoğlu, Berrin and Buntine, Wray},
  volume = 	 {222},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {11--14 Nov},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v222/baykal24a/baykal24a.pdf},
  url = 	 {https://proceedings.mlr.press/v222/baykal24a.html},
  abstract = 	 {Diffusion models are generative models that have shown significant advantages compared to other generative models in terms of higher generation quality and more stable training. However, the computational need for training diffusion models is considerably increased. In this work, we incorporate prototype learning into diffusion models to achieve high generation quality faster than the original diffusion model. Instead of randomly initialized class embeddings, we use separately learned class prototypes as the conditioning information to guide the diffusion process. We observe that our method, called ProtoDiffusion, achieves better performance in the early stages of training compared to the baseline method, signifying that using the learned prototypes shortens the training time. We demonstrate the performance of ProtoDiffusion using various datasets and experimental settings, achieving the best performance in shorter times across all settings.}
}
```
