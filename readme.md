# ProvNeRF: Modeling per Point Provenance in NeRFs as a Stochastic Field
**[ProvNeRF: Modeling per Point Provenance in NeRFs as a Stochastic Field](https://provnerf.github.io/)**

Kiyohiro Nakayama, Mikaela Angelina Uy, Yang You, Ke Li, Leonidas Guibas

NeurIPS 2024

![Alt text](assets/teaser.jpg)

## Introduction
Neural radiance fields (NeRFs) have gained popularity with multiple works showing promising results across various applications. However, to the best of our knowledge, existing works do not explicitly model the distribution of training camera poses, or consequently the triangulation quality, a key factor affecting reconstruction quality dating back to classical vision literature. We close this gap with ProvNeRF, an approach that models the provenance for each point -- i.e., the locations where it is likely visible -- of NeRFs as a stochastic field. We achieve this by extending implicit maximum likelihood estimation (IMLE) to functional space with an optimizable objective. We show that modeling per-point provenance during the NeRF optimization enriches the model with information on triangulation leading to improvements in novel view synthesis and uncertainty estimation under the challenging sparse, unconstrained view setting against competitive baselines.


## Pretrained Models
ProvNeRF pretrained models can be downloaded [here](http://download.cs.stanford.edu/orion/ProvNeRF/provnerf_pretrained.zip). Place the unzipped folder at the root of the repository.
## Data
We use the processed Scannet and Tanks and Temples datasets from [SCADE](http://download.cs.stanford.edu/orion/scade/datasets.zip). Please follow the instructions there to download and unzip the data. Place the unzipped folder at the root of the repository.
## Code

### Environment Set-up
```bash
# first clone the repo by 
git clone https://github.com/diffFacto/diffFacto.git && cd diffFacto
# Create conda environment with python 3.8
conda create -n provnerf python=3.8 -y
conda activate provnerf
# we use torch 2.0.0 and CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# we insall other dependencies
pip install tqdm tensorboardX opencv-python configargparse lpips pathlib tensorboard scikit-image 
```
### Training ProvNeRF for NVS regularization
To train ProvNeRF for NVS regularization, we need to first train the provenence stochastic field using fIMLE. We provide the trained checkpoints in the `provnerf_pretrained/fimle` directory. To train a new provenance stochastic field for different scenes on top of a SCADE pretrained model, run the command in `provnerf_pretrained/fimle/[scannet/tnt]/[scene_id]/script.sh`.

After the provenance stochastic field is trained, we can improve SCADE's NVS quality by running our NVS regularizer. We provide checkpoints after NVS regualization in the `provnerf_pretrained/nvs_ckpts` directory. To render the test views, run the testing command given in To train a new model with NVS regularization on top of a SCADE pretrained model, run the command in `provnerf_pretrained/nvs_ckpts/[scannet/tnt]/[scene_id]/script.sh`. To regularize the NVS from a SCADE checkpoint, run the training command in the same script.


## License
This repository is released under MIT License (see LICENSE file for details).

## Citation
```
@inproceedings{nakayama2024provnerf,
    title={ProvNeRF: Modeling per Point Provenance in NeRFs as a Stochastic Field}, 
    author={Kiyohiro Nakayama and Mikaela Angelina Uy and Yang You and Ke Li and Leonidas Guibas},
    journal = {Advances in Neural Information Processing Systems (NeurIPS)}},
    year={2024}
}
```