# Automatic-Lipreading-translator
This page serves as an landing page of the project, describing the project for DT2119 at KTH . For more information, go to the [wiki](https://github.com/LuukEbenau/Automatic-Lipreading-translator/wiki). 
# Code writing instruction
For each sub-feature of our program


## Requirements
The required python modules can be installed by running the following command from the top folder of this repository
```
pip install -r requirements.txt
```
# Run the code using
```
python train.py --data <Absolute path to location of Automatic-Lipreading-translator>/data/GRID --data_name GRID --gpu 0 --workers <number of cpu threads> --lr 0.0001 --epochs 200 --batch_size <selected batch size> --checkpoint_dir <absolute path to folder of checkpoints> --output_content_loss --asr_checkpoint "openai/whisper-tiny.en" --asr_checkpoint_type WHISPER --eval_step 400 --visual_front_checkpoint <absolute path to location of Automatic-Lipreading-translator>/pretrained/LRS2_front_PT.ckpt 
```

### Datasets
The GRID Audio-Visual Speech Corpus can be downloaded from:
- https://zenodo.org/records/3625687/



#### Pretrain-Trainval-Test Split
Our recommended split is found below and is divided into 1 low-error and 1 high-error speaker of both genders in trainval and test, and 1 mid-error male speaker in trainval and test.
This results in gender-equal dataset pool for training where 2000 random samples are picked from at each epoch.


<div align="center">

| Split    | Speakers                           |
|:--------:|:-----------------------------------:|
| Trainval | s8 (M), s19 (M), s23 (F), s31 (F), s32 (M) |
| Test     | s13 (M), s14 (M), s17 (M), s24 (F), s33 (F) |

</div>


#### Known issues in dataset
As of June 23rd 2024, one speaker is missing (s21) and the following alignments are misplaced:

<div align="center">

| Split | Speakers | Alignments found in |
|:-------:|:----------:|:---------------------:|
| TRAINVAL | S19<br>S23<br>S32 | S18<br>S22<br>S33 |
| TEST | S13<br>S14<br>S33 | S12<br>S15<br>S32 |
| PRETRAIN | S10<br>S11<br>S12<br>S14<br>S18<br>S20<br>S22<br>S26<br>S27<br>S28<br>S29<br>S5<br>S6 | S13<br>S10<br>S11<br>S15<br>S19<br>S21<br>S23<br>S27<br>S26<br>S29<br>S28<br>S6<br>S5 |

</div>

## Architecture

**Encoder:** Utilizes a pre-trained _ResNet-18_ and _Conformer_ to embed visual features from mouth crops, along with a one-dimensional CNN for embedding mel-spectrogram snippets during training.

**Decoder:** Generates high-quality mel-spectrograms using a 1D convolutional network conditioned on visual and speaker embeddings, with _adaptive layer normalization(AdaLn)_.

**ASR**: Unlike the original model which used an ASR trained specifically on the LRS2 dataset, this model uses a pre-trained _Whisper-tiny_ model for better adaptability to new datasets.

**Vocoder:** Employs the _HiFi-GAN_ for efficient and high-quality speech synthesis from generated mel-spectrograms.


<div align="center">
  <img src="https://github.com/LuukEbenau/Automatic-Lipreading-translator/blob/main/imgs/Modified_Lip-to-Speech_Architecture.png" width="500" alt="Modified Lip-to-Speech Architecture">
</div>

## Training and Evaluation

### Loss Function
- **Combined loss:** $L_{tot} = λ_{ctc} * L_{ctc} + λ_{rec} * L_{rec} + λ_{asr} * L_{asr}$
- **Components:** CTC-loss ($L_{ctc}$), Reconstruction-loss ($L_{rec}$), ASR-loss ($L_{asr}$)
- Optimizes: Text alignment, mel-spectrogram fidelity, word prediction accuracy

### Evaluation Metrics
- **Quantitative:** STOI, ESTOI, WER

### Experiment Setup
- Platform: Google Cloud
- Dataset: GRID
- Training: 2000 samples/epoch, batch size 16, 3 speakers
- Model: Learning rate 0.00001, AdamW optimizer
- Hardware: NVIDIA® L4 GPU
- Training time: ~12 hours with ASR component


## Evaluation Results

### Performance Comparison

<div align="center">
  
| Model | STOI | ESTOI | WER* |
|:------|:----:|:-----:|:----:|
| $\text{Our Model}_{GRID}$  | 0.54 | 0.27 | 70% |
| $\text{Our Model}_{GRID}$ + ASR | 0.481 | 0.203 | 76% |
| $Model_{LRS2}$ | 0.526 | 0.341 | 60% |

</div>

*Actual WER values are  lower than 50% since ther was an issue with extra `SIL` tokens at start and end causing shifting of alignments hence artificially higher WER.

### Discussion

#### ASR-loss Behavior
During training, we observed that the ASR-loss occasionally dropped to 0, which suggests potential issues in our ASR-loss calculations.

#### Reconstruction Loss
The reconstruction loss exhibited a more gradual convergence when incorporating the ASR-loss, indicating a potentially more stable backpropagation process.

#### ASR-loss Impact
This leads us to believe that the inclusion of ASR-loss likely provides a more diverse and accurate gradient calculation.

#### WER Analysis
Our analysis of the Word Error Rate (WER) reveals that the predicted text quality is better than the WER suggests. The inflated error rates can be attributed to repeated 'SIL' tokens for silence and extra unknown tokens in the ground truth. When conducting local tests without considering silence or character mismatches, we achieved a WER of approximately 50%.

#### Model Comparison
In comparison to the original LRS2 model, our model shows competitive STOI scores. However, our ESTOI scores are lower than those of the original LRS2 model. It's worth noting that our ASR model demonstrates a higher WER, which we attribute to the fact that we didn't fine-tune it, unlike the reported results for LRS2.


## Documentation & Communication
We have not discussed it yet in detail, but what we did discuss is to make primarily use of github for the means of documentation and communication. We will use this Readme.MD for general information, the Getting Started, Etc. For anything more detailed, we can use the github Wiki. Just click on [wiki](https://github.com/LuukEbenau/Automatic-Lipreading-translator/wiki) in the header, and you can add pages for each of the subjects. We can for example make a page for each of the modules in the code, 1 page about architecture, etc. By putting it all in the github we make sure that everyone has access to the right information, and additionally it will safe us a lot of work for the final report.
## Code & Source Control
One way of working which I usually like is having feat/ feature branches for features which dont work yet, development branch for partial work and stable code on the main branch. Based on what you guys prefer we can do mandatory code reviews & pull requests for pushing to the main branch, But i'm also okay with ommiting this, and just bearing the responsibility of not pushing broken code to the main branch :) Lets discuss about these things in the next meeting(s). 


# Links

1. [Interesting: VIPL-Audio-Visual-Speech-Understanding](https://github.com/VIPL-Audio-Visual-Speech-Understanding)
2. [Research paper: IEEE Xplore](https://ieeexplore.ieee.org/document/9272286)
3. [Learning Individual Speaking Styles for Accurate Lip to Speech Synthesis (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Prajwal_Learning_Individual_Speaking_Styles_for_Accurate_Lip_to_Speech_Synthesis_CVPR_2020_paper.pdf)
   - [GitHub: Lip2Wav](https://github.com/Rudrabha/Lip2Wav)
4. [Face extraction: MediaPipe Solutions Guide](https://ai.google.dev/edge/mediapipe/solutions/guide)

# Notes
What are we going to do?
We are going to use the research of [LIP-TO-SPEECH SYNTHESIS IN THE WILD WITH MULTI-TASK LEARNING](https://arxiv.org/pdf/2302.08841.pdf). Our goal will be to improve this paper in a variety of ways.
The first way of improvement will be to to replace part of the algorithm with a random algorithm using LMM's which orders the words
