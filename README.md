# <p align=center> :fire: `TAR3D: Creating High-Quality 3D Assets via Next-Part Prediction`</p>


![framework_img](assets/method.png)
<div align="center">
  
  [[Paper](https://arxiv.org/pdf/2412.16919)] &emsp; [[Project Page](https://zhangxuying1004.github.io/projects/TAR3D/)] &emsp;  [[Jittor Version]()]&emsp; [[Demo]()]   <br>

</div>


## 🚀 Efficient Architecture Updates
This repository has been updated with:
1.  **Hybrid Attention**: Interleaves efficient Linear Attention (Lightning Attention) with standard Full Attention (Flash Attention) in a 3:1 ratio. This reduces memory complexity from $O(N^2)$ to $O(N)$ for most layers.
2.  **Muon Optimizer**: Uses the memory-efficient Muon optimizer for weight updates, enabling larger batch sizes.
3.  **Single-GPU Support**: Optimized to run on consumer GPUs without requiring a massive cluster.

### Additional Dependencies
Please install the following efficient attention kernels:
```bash
pip install lightning-attn flash-attn --no-build-isolation
```

> **Note**: This single-GPU optimization integrates efficient training techniques into the conditional [TAR3D](https://github.com/HVision-NKU/TAR3D) framework, inspired by the efficient architecture of the unconditional [iFlame](https://github.com/hanxiaowang00/iFlame) model. I adapted the Linear Attention mechanism to correctly handle conditional generation (handling padding without explicit masks) while retaining the original model's quality.

## 🚩 **Todo List**
- [x] Source code of 3D VQVAE.
- [x] Source code of 3D GPT.
- [x] Source code of 3D evaluation.
- [x] 10w uids of high-quality objaverse object.
- [ ] Pretrained weights of 3D reconstruction.
- [ ] Pretrained weights of image-to-3D generation.
- [ ] Pretrained weights of text-to-3D generation.


## :books: BibTeX
If you find TAR3D useful for your research or applications, please give us a star and cite this paper:

```BibTeX
@inproceedings{zhang2025tar3d,
  title={Tar3d: Creating high-quality 3d assets via next-part prediction},
  author={Zhang, Xuying and Liu, Yutong and Li, Yangguang and Zhang, Renrui and Liu, Yufei and Wang, Kai and Ouyang, Wanli and Xiong, Zhiwei and Gao, Peng and Hou, Qibin and Cheng, Ming-Ming},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5134--5145},
  year={2025}
}
```

## ⚙️ Setup
### 1. Dependencies and Installation
We recommend using `Python>=3.10`, `PyTorch>=2.1.0`, and `CUDA>=12.1`.
```bash
conda create --name tar3d python=3.10
conda activate tar3d
pip install -U pip

# Ensure Ninja is installed
conda install Ninja

# Install the correct version of CUDA
conda install cuda -c nvidia/label/cuda-12.1.0

# Install PyTorch and xformers
# You may need to install another xformers version if you use a different PyTorch version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post7

# For Linux users: Install Triton 
pip install triton

# Install other requirements
pip install -r requirements.txt
```
### 2. Downloading Datasets
- [ShapeNetV2](https://drive.google.com/drive/folders/1UFPi_UklH5clWKxxeL1IsxfjdUfc7i4x)  
- [Objaverse](https://huggingface.co/datasets/allenai/objaverse)   
- [ULIP](https://huggingface.co/datasets/SFXX/ulip/tree/main)  
- [Objaverse_high_quality_uids(10w)](https://raw.githubusercontent.com/HVision-NKU/TAR3D/refs/heads/main/Objaverse_high_quality_uids.txt)

### 3. Downloading Checkpoints
We are currently unable to access the ckpts stored on the aliyun space used during the internship.  
We will retrain a version as soon as possible.


## ⚡ Quick Start

### 1. Reconstructing a 3D Geometry with 3D VQ-VAE
```
python infer_vqvae.py
```

### 2. Conditional 3D Generation
```
python run.py --gpt-type i23d
```


## 💻 Training
### 1. Training 3D VQ-VAE
```
python train_vqvae.py --base configs/vqvae3d.yaml --gpus 0,1,2,3,4,5,6,7 --num_nodes 1
```
In practice, we first train the encoder and decoder of our VQ-VAE according to the scheme of VAE.   
Then, we add the vector quantization codebook and fine-tune the entire VQ-VAE.


### 2. Training 3D GPT
#### Single-GPU Efficient Training (New!)
Thanks to the new Hybrid Linear Attention architecture and Muon optimizer, you can now train efficiently on a single GPU.
```bash
python train_gpt.py \
--gpt-type i23d \
--global-batch-size 4 \
--gpt-model GPT-B
```

#### Multi-GPU Training (Legacy)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
--nnodes=1 \
--nproc_per_node=8 \
--node_rank=0 \
--master_addr='127.0.0.1' \
--master_port=29504 \
train_gpt.py \
--gpt-type i23d \
--global-batch-size 8 "$@"

```



## 💫 Evaluation
### 1. 2D Evaluation (PSNR, SSIM, Clip-Score, LPIPS)
```
python eval_2d.py
```

### 2. 3D Evaluation (Chamfer Distance, F-Score)
```
python eval_3d.py
```


## 🤗 Acknowledgements

We thank the authors of the following projects for their excellent contributions to 3D generative AI!

- [LlamaGen](https://github.com/FoundationVision/LlamaGen/)
- [Michelangelo](https://github.com/NeuralCarver/Michelangelo/)
- [InstantMesh](https://github.com/TencentARC/InstantMesh)
- [OpenLRM](https://github.com/3DTopia/OpenLRM)
- [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet)



### Additional Dependencies
Please install the following efficient attention kernels:
```bash
pip install lightning-attn flash-attn --no-build-isolation
```

> **Note**: This single-GPU optimization integrates efficient training techniques into the conditional [TAR3D](https://github.com/HVision-NKU/TAR3D) framework, inspired by the efficient architecture of the unconditional [iFlame](https://github.com/hanxiaowang00/iFlame) model. I adapted the Linear Attention mechanism to correctly handle conditional generation (handling padding without explicit masks) while retaining the original model's quality.

## 🚩 **Todo List**
- [x] Source code of 3D VQVAE.
- [x] Source code of 3D GPT.
- [x] Source code of 3D evaluation.
- [x] 10w uids of high-quality objaverse object.
- [ ] Pretrained weights of 3D reconstruction.
- [ ] Pretrained weights of image-to-3D generation.
- [ ] Pretrained weights of text-to-3D generation.


## :books: BibTeX
If you find TAR3D useful for your research or applications, please give us a star and cite this paper:

```BibTeX
@inproceedings{zhang2025tar3d,
  title={Tar3d: Creating high-quality 3d assets via next-part prediction},
  author={Zhang, Xuying and Liu, Yutong and Li, Yangguang and Zhang, Renrui and Liu, Yufei and Wang, Kai and Ouyang, Wanli and Xiong, Zhiwei and Gao, Peng and Hou, Qibin and Cheng, Ming-Ming},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5134--5145},
  year={2025}
}
```

## ⚙️ Setup
### 1. Dependencies and Installation
We recommend using `Python>=3.10`, `PyTorch>=2.1.0`, and `CUDA>=12.1`.
```bash
conda create --name tar3d python=3.10
conda activate tar3d
pip install -U pip

# Ensure Ninja is installed
conda install Ninja

# Install the correct version of CUDA
conda install cuda -c nvidia/label/cuda-12.1.0

# Install PyTorch and xformers
# You may need to install another xformers version if you use a different PyTorch version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post7

# For Linux users: Install Triton 
pip install triton

# Install other requirements
pip install -r requirements.txt
```
### 2. Downloading Datasets
- [ShapeNetV2](https://drive.google.com/drive/folders/1UFPi_UklH5clWKxxeL1IsxfjdUfc7i4x)  
- [Objaverse](https://huggingface.co/datasets/allenai/objaverse)   
- [ULIP](https://huggingface.co/datasets/SFXX/ulip/tree/main)  
- [Objaverse_high_quality_uids(10w)](https://raw.githubusercontent.com/HVision-NKU/TAR3D/refs/heads/main/Objaverse_high_quality_uids.txt)

### 3. Downloading Checkpoints
We are currently unable to access the ckpts stored on the aliyun space used during the internship.  
We will retrain a version as soon as possible.


## ⚡ Quick Start

### 1. Reconstructing a 3D Geometry with 3D VQ-VAE
```
python infer_vqvae.py
```

### 2. Conditional 3D Generation
```
python run.py --gpt-type i23d
```


## 💻 Training
### 1. Training 3D VQ-VAE
```
python train_vqvae.py --base configs/vqvae3d.yaml --gpus 0,1,2,3,4,5,6,7 --num_nodes 1
```
In practice, we first train the encoder and decoder of our VQ-VAE according to the scheme of VAE.   
Then, we add the vector quantization codebook and fine-tune the entire VQ-VAE.


### 2. Training 3D GPT
#### Single-GPU Efficient Training (New!)
Thanks to the new Hybrid Linear Attention architecture and Muon optimizer, you can now train efficiently on a single GPU.
```bash
python train_gpt.py \
--gpt-type i23d \
--global-batch-size 4 \
--gpt-model GPT-B
```

#### Multi-GPU Training (Legacy)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
--nnodes=1 \
--nproc_per_node=8 \
--node_rank=0 \
--master_addr='127.0.0.1' \
--master_port=29504 \
train_gpt.py \
--gpt-type i23d \
--global-batch-size 8 "$@"

```



## 💫 Evaluation
### 1. 2D Evaluation (PSNR, SSIM, Clip-Score, LPIPS)
```
python eval_2d.py
```

### 2. 3D Evaluation (Chamfer Distance, F-Score)
```
python eval_3d.py
```


## 🤗 Acknowledgements

We thank the authors of the following projects for their excellent contributions to 3D generative AI!

- [LlamaGen](https://github.com/FoundationVision/LlamaGen/)
- [Michelangelo](https://github.com/NeuralCarver/Michelangelo/)
- [InstantMesh](https://github.com/TencentARC/InstantMesh)
- [OpenLRM](https://github.com/3DTopia/OpenLRM)
- [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet)



Implementation Note: This optimization integrates techniques from unconditional models (like iFlame) into the conditional TAR3D framework. I have adapted the Linear Attention mechanism to correctly handle conditional generation (variable-length sequences without explicit masks).🚩 Todo List$$x$$ Source code of 3D VQ-VAE.$$x$$ Source code of 3D GPT (Optimized).$$x$$ Source code of 3D evaluation.$$x$$ 100k UIDs of high-quality Objaverse objects.$$ $$ Pretrained weights of 3D reconstruction.$$ $$ Pretrained weights of image-to-3D generation.$$ $$ Pretrained weights of text-to-3D generation.📚 BibTeXThis repository is based on the original TAR3D research. If you use this code, please cite the original paper:@inproceedings{zhang2025tar3d,
  title={Tar3d: Creating high-quality 3d assets via next-part prediction},
  author={Zhang, Xuying and Liu, Yutong and Li, Yangguang and Zhang, Renrui and Liu, Yufei and Wang, Kai and Ouyang, Wanli and Xiong, Zhiwei and Gao, Peng and Hou, Qibin and Cheng, Ming-Ming},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5134--5145},
  year={2025}
}

⚙️ Setup1. Dependencies and InstallationWe recommend using Python>=3.10, PyTorch>=2.1.0, and CUDA>=12.1.conda create --name tar3d-eff python=3.10
conda activate tar3d-eff
pip install -U pip

# Ensure Ninja is installed
conda install Ninja

# Install the correct version of CUDA
conda install cuda -c nvidia/label/cuda-12.1.0

# Install PyTorch and xformers
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install xformers==0.0.22.post7

# For Linux users: Install Triton 
pip install triton

# Install other requirements
pip install -r requirements.txt

2. Downloading DatasetsShapeNetV2ObjaverseULIPObjaverse_high_quality_uids(10w)3. CheckpointsNote: Original checkpoints are currently unavailable. We will release retrained weights for this efficient version soon.⚡ Quick Start1. Reconstructing a 3D Geometry with VQ-VAEpython infer_vqvae.py

2. Conditional 3D Generationpython run.py --gpt-type i23d

💻 Training1. Training 3D VQ-VAEpython train_vqvae.py --base configs/vqvae3d.yaml --gpus 0,1,2,3,4,5,6,7 --num_nodes 1

Note: Training first focuses on the encoder/decoder (VAE scheme), then fine-tunes with the vector quantization codebook.2. Training 3D GPTSingle-GPU Efficient Training (New!)Thanks to the Hybrid Linear Attention architecture and Muon optimizer, you can now train efficiently on a single GPU.python train_gpt.py \
--gpt-type i23d \
--global-batch-size 4 \
--gpt-model GPT-B

Multi-GPU Training (Legacy Support)CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
--nnodes=1 \
--nproc_per_node=8 \
--node_rank=0 \
--master_addr='127.0.0.1' \
--master_port=29504 \
train_gpt.py \
--gpt-type i23d \
--global-batch-size 8 "$@"

💫 Evaluation1. 2D Evaluation (PSNR, SSIM, Clip-Score, LPIPS)python eval_2d.py

2. 3D Evaluation (Chamfer Distance, F-Score)python eval_3d.py

🤗 AcknowledgementsThis efficient implementation builds upon the excellent research of the original TAR3D authors. We also acknowledge:LlamaGenMichelangeloInstantMeshOpenLRM3DShape2VecSet
