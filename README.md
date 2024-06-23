# Automatic-Lipreading-translator
This page serves as an landing page of the project, describing the project globally. For more information, go to the [wiki](https://github.com/LuukEbenau/Automatic-Lipreading-translator/wiki). 
# Code writing instruction
For each sub-feature of our program


# run the code using
```
python train.py --data <Absolute path to location of Automatic-Lipreading-translator>/data/GRID --data_name GRID --gpu 0 --workers <number of cpu threads> --lr 0.0001 --epochs 200 --batch_size <selected batch size> --checkpoint_dir <absolute path to folder of checkpoints> --output_content_loss --asr_checkpoint "openai/whisper-tiny.en" --asr_checkpoint_type WHISPER --eval_step 400 --visual_front_checkpoint <absolute path to location of Automatic-Lipreading-translator>/pretrained/LRS2_front_PT.ckpt 
```

## Requirements
The required python modules can be installed by running the following command from the top folder of this repository
```
pip install -r requirements.txt
```

### Datasets
#### Download
The GRID Audio-Visual Speech Corpus can be downloaded from:
- https://zenodo.org/records/3625687/

#### Pretrain-Trainval-Test Split
Our recommended split is found below and is divided into 1 low-error and 1 high-error speaker of both genders in trainval and test, and 1 mid-error male speaker in trainval and test.
This results in gender-equal dataset pool for training where 2000 random samples are picked from at each epoch.

##### Trainval
s8 (M), s19 (M), s23 (F), s31 (F), s32 (M)

##### Test
s13 (M), s14 (M), s17 (M), s24 (F). s33 (F)

#### Known issues in dataset
As of June 23rd 2024, one speaker is missing (s21) and the following alignments are misplaced:

Split location		Speaker		Alignments found in
TRAINVAL		S19		S18
TRAINVAL		S23		S22
TRAINVAL		S32		S33
TEST			S13		S12
TEST			S14		S15
TEST			S33		S32
PRETRAIN		S10		S13
PRETRAIN		S11		S10
PRETRAIN		S12		S11
PRETRAIN		S14		S15
PRETRAIN		S18		S19
PRETRAIN		S20		S21
PRETRAIN		S22		S23
PRETRAIN		S26		S27
PRETRAIN		S28		S29
PRETRAIN		S29		S28
PRETRAIN		S5		S6
PRETRAIN		S6		S5


## Documentation & Communication
We have not discussed it yet in detail, but what we did discuss is to make primarily use of github for the means of documentation and communication. We will use this Readme.MD for general information, the Getting Started, Etc. For anything more detailed, we can use the github Wiki. Just click on [wiki](https://github.com/LuukEbenau/Automatic-Lipreading-translator/wiki) in the header, and you can add pages for each of the subjects. We can for example make a page for each of the modules in the code, 1 page about architecture, etc. By putting it all in the github we make sure that everyone has access to the right information, and additionally it will safe us a lot of work for the final report.
## Code & Source Control
One way of working which I usually like is having feat/ feature branches for features which dont work yet, development branch for partial work and stable code on the main branch. Based on what you guys prefer we can do mandatory code reviews & pull requests for pushing to the main branch, But i'm also okay with ommiting this, and just bearing the responsibility of not pushing broken code to the main branch :) Lets discuss about these things in the next meeting(s). 


# Links
1. Intesting: https://github.com/VIPL-Audio-Visual-Speech-Understanding
2. research paper: https://ieeexplore.ieee.org/document/9272286
3. https://openaccess.thecvf.com/content_CVPR_2020/papers/Prajwal_Learning_Individual_Speaking_Styles_for_Accurate_Lip_to_Speech_Synthesis_CVPR_2020_paper.pdf https://github.com/Rudrabha/Lip2Wav
4. Face extracting: https://ai.google.dev/edge/mediapipe/solutions/guide

# Notes
What are we going to do?
We are going to use the research of [LIP-TO-SPEECH SYNTHESIS IN THE WILD WITH MULTI-TASK LEARNING](https://arxiv.org/pdf/2302.08841.pdf). Our goal will be to improve this paper in a variety of ways.
The first way of improvement will be to to replace part of the algorithm with a random algorithm using LMM's which orders the words
