# BEP_VIT

Swin transformer For rigid image registration.

## Requirements
To use this code you will need to create an Anaconda virtual environment which should be installed with Pytorch and other required libraries.

<pre><code>conda create -y -n pytorch_VIT
conda activate pytorch_VIT
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install scikit-image numpy
conda install -c simpleitk simpleitk
conda install -c conda-forge matplotlib tqdm
conda install -c conda-forge timm
conda install -c conda-forge einops
pip install ml-collections</code></pre>
  
## Source code 
This project uses source code from:
1. Swin-Transformer code retrieved from https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation
2. TransMorph https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration.git
