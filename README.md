<p align=center> :fire: <b>TAR3D-Efficient:</b> Optimized High-Quality 3D Assets via Next-Part Prediction</p><div align="center">[Original Paper]   [Original Project Page]   [This Efficient Fork]   [Demo] </div>⚠️ Fork Notice: Efficient ImplementationThis repository is an unofficial, efficiency-focused fork of TAR3D.It introduces architectural changes to enable training on consumer hardware while maintaining generation quality. This is not the official implementation.🚀 Efficient Architecture UpdatesThis fork introduces:Hybrid Attention Mechanism: Interleaves efficient Linear Attention (Lightning Attention) with standard Full Attention (Flash Attention) in a 3:1 ratio. This reduces memory complexity from $O(N^2)$ to $O(N)$ for the majority of layers.Muon Optimizer: Integrated the memory-efficient Muon optimizer for weight updates, allowing for significantly larger batch sizes on limited VRAM.True Single-GPU Support: The training loop has been optimized to run fully on a single consumer GPU (e.g., RTX 3090/4090) without requiring a massive cluster.Additional DependenciesTo use this efficient version, please install the required kernels:pip install lightning-attn flash-attn --no-build-isolation
Implementation Note: This optimization integrates techniques from unconditional models (like iFlame) into the conditional TAR3D framework. I have adapted the Linear Attention mechanism to correctly handle conditional generation (variable-length sequences without explicit masks).🚩 Todo List[x] Source code of 3D VQ-VAE.[x] Source code of 3D GPT (Optimized).[x] Source code of 3D evaluation.[x] 100k UIDs of high-quality Objaverse objects.[ ] Pretrained weights of 3D reconstruction.[ ] Pretrained weights of image-to-3D generation.[ ] Pretrained weights of text-to-3D generation.📚 BibTeXThis repository is based on the original TAR3D research. If you use this code, please cite the original paper:@inproceedings{zhang2025tar3d,
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