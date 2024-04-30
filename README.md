# Unlocking Voices, Unleashing Possibilities: Your Words, Our Recognition!

![Made With python 3.11.5](https://img.shields.io/badge/Made%20with-Python%203.11.5-brightgreen)![pytorch](https://img.shields.io/badge/Made%20with-pytorch-green.svg)![librosa](https://img.shields.io/badge/Made_with-librosa-blue)![speechbrain](https://img.shields.io/badge/Made_with-speechbrain-brown)![huggingface](https://img.shields.io/badge/Made_with-huggingface-violet)


### Code:

Below are the step to setup the code and perform training

### Setup:

After setting up the code as below, update the paths appropriately

> git clone https://github.com/ksasi/asr.git
> 
> cd asr
> 
> git clone https://github.com/speechbrain/speechbrain.git
> 
> 

### Install Dependencies:

> cd asr/speechbrain
> 
> pip install -r requirements.txt
> 
> cd ..
> 
> pip install -r requirements.txt
> 

- copy all the files from `<root_path>/code` to `<root_path>/speechbrain/recipes/LibriSpeech/ASR/transformer`

### Datasets :

- Create a directory named ***datasets*** under ***asr***
- Download [LibriSpeech] (https://www.openslr.org/12) (specifically train-clean-100, test-clean and dev-clean partitions)

Execute the below steps to generate **Noisy LibriSpeech** dataset

>cd code
>
>nohup python generate\_noisy\_librispeech.py --wham\_dir \<root\_path\>/datasets/wham\_noise --libri\_dir \<root\_path\>/datasets/LibriSpeech >> \<root\_path\>/logs/librispeech\_wham\_noise.out &
>
>

### Models Evaluation (Using **Noisy LibriSpeech**)

#### Evaluation of pretrained **wav2vec2** and **Conformer** (pretrained on original Librispeech)

- [**wav2vec2**](https://huggingface.co/speechbrain/asr-wav2vec2-librispeech) : speechbrain/asr-wav2vec2-librispeech
- [**Conformer**](https://huggingface.co/speechbrain/asr-conformer-transformerlm-librispeech) : speechbrain/asr-conformer-transformerlm-librispeech

>cd code
>
>nohup python model\_eval.py --libri\_dir \<root\_path\>/datasets/LibriSpeech/test-clean >> \<root\_path\>/logs/eval\_wav2vec2\_conformer.log &
>
>

#### Evaluation of pretrained **Branchformer** (pretrained on original Librispeech)

- [**branchformer**](https://huggingface.co/pyf98/librispeech_100_ctc_e_branchformer) : pyf98/librispeech_100_ctc_e_branchformer

>cd code
>
>nohup python branchformer\_eval.py --libri\_dir \<root\_path\>/datasets/LibriSpeech/test-clean >> \<root\_path\>/logs/eval\_branchformer.log &
>
>

#### Fine-tune Conformer (on **Noisy LibriSpeech**)

>cd \<root\_path\>/speechbrain/recipes/LibriSpeech/ASR/transformer
>
>nohup python finetune\_conformer.py \<root\_path\>/hparams/conformer\_large.yaml --data_folder=\<root\_path\>/datasets/LibriSpeech >> \<root\_path\>/logs/finetune\_conformer.out &

#### Fine-tune TSConformer (on **Noisy LibriSpeech**)

- **TSConformer** is a Custom Conformer Architecture using [**TaylorSeries Linear Attention**](https://arxiv.org/abs/2312.04927)

>cd \<root\_path\>/speechbrain/recipes/LibriSpeech/ASR/transformer
>
>nohup python finetune\_tsconformer.py \<root\_path\>/hparams/tsconformer\_large.yaml --data\_folder=\<root\_path\>/datasets/LibriSpeech >> \<root\_path\>/logs/finetune\_tsconformer.out &
>


### Demo 

Demo of **Automatic Speech Recognition on Noisy Data** can be executed by running `ASR_Demo.ipynb` ipython notebook in the demo folder




### References

- Speechbrain - [Github Link](https://github.com/speechbrain/speechbrain/tree/develop/recipes/WSJ0Mix/separation)
- EER Metric - [blog](https://yangcha.github.io/EER-ROC/)
- SepFormer Huggingface - [Link](https://huggingface.co/speechbrain/sepformer-whamr)
- Torchmetrics - [Link](https://lightning.ai/docs/torchmetrics/stable/audio/scale_invariant_signal_noise_ratio.html)
- branchformer - [Link](https://huggingface.co/pyf98/librispeech_100_ctc_e_branchformer)
- wav2vec2 - [Link](https://huggingface.co/speechbrain/asr-wav2vec2-librispeech)
- Conformer - [Link](https://huggingface.co/speechbrain/asr-conformer-transformerlm-librispeech)
- Taylor Series Linear Attention - [Link](https://github.com/lucidrains/taylor-series-linear-attention)
