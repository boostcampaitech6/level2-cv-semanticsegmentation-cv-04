# AI Tech 6기 Team 아웃라이어

## Members
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/kangshwan">
        <img src="https://imgur.com/ozd1yor.jpg" width="100" height="100" /><br>
        강승환
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/viitamin">
        <img src="https://imgur.com/GXteBDS.jpg" width="100" height="100" /><br>
        김승민
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/tjfgns6043">
        <img src="https://imgur.com/aMVcwCF.jpg" width="100" height="100" /><br>
        설훈
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/leedohyeong">
        <img src="https://imgur.com/F6ZfcEl.jpg" width="100" height="100" /><br>
        이도형
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/wjsqudrhks">
        <img src="https://imgur.com/ZSVCV82.jpg" width="100" height="100" /><br>
        전병관
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/seonghyeokcho">
        <img src="https://imgur.com/GBdY0k4.jpg" width="100" height="100" /><br>
        조성혁
      </a>
    </td>
  </tr>
</table>

## Hand Bone Image Segmentation

뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치기 때문에, 정확한 뼈 분할은 의료 진단 및 치료 계획을 개발하는 데 필수적입니다.

Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 특히, 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며, 다양한 목적으로 도움을 줄 수 있습니다.

이번 프로젝트는 부스트캠프 AI Tech CV 트랙내에서 진행된 대회이며 dice score로 최종평가를 진행하게 됩니다

## Final Score
Public

![Public](https://imgur.com/GoRttFS.jpg)

Private

![private](https://imgur.com/Lpd4KAc.jpg)

## Wrap-Up-Report
[Wrap-Up-Report](https://synonymous-ton-89f.notion.site/Wrap-up-Reports-34b15334503c43f2ad4ad716c972fb81?pvs=4)

## Ground Rules
### [Conventional Commits 1.0.0](https://www.conventionalcommits.org/ko/v1.0.0/)
```bash
<타입>[적용 범위(선택 사항)]: <설명>

[본문(선택 사항)]

[꼬리말(선택 사항)]
```

#### Types
- fix | feat | BREAKING CHANGE | build | chore | ci | docs | style | refactor | test | release
  - fix : 기능에 대한 버그 수정
  - feat : 새로운 기능 추가, 기존의 기능을 요구 사항에 맞추어 수정
  - build : 빌드 관련 수정
  - chore : 패키지 매니저 수정, 그 외 기타 수정 ex) .gitignore
  - ci : CI 관련 설정 수정
  - docs : 문서(주석) 수정
  - style : 코드 스타일, 포맷팅에 대한 수정
  - refactor : 기능의 변화가 아닌 코드 리팩터링 ex) 변수 이름 변경
  - test : 테스트 코드 추가/수정
  - release : 버전 릴리즈

## Requirements
* Python >= 3.10.13
* PyTorch >= 1.12.1
* mmcv-full >= 1.6.2

## Folder Structure
  ```
code
├── mmsegmentation
│   ├── configs
|   ├── mmseg
│   ├── inference.ipynb
│   └── requirements.txt
│
├── utils
│   ├── hard_voting_ensemble.ipynb
│   └── visualization.ipynb
├── dataset.py
├── inference.py
└── main.py
  ```
## Dataset
- Total Images : 1088장 (train : 800, test : 288)
- 29 Class
- Image Size : (2048, 2048)

## ETC
- 추후 Swin-Unet모델을 올려놓겠습니다. 실험을 많이 진행하지 못했지만 1024,1024크기로 실험을 진행하면 좋은 결과가 있을 것으로 매우매우 예상됩니다.
- 다음 기수분들이 꿈을 이뤄주세요. 모두가 Unet++를 쓰고 있습니다. 그리고 mmseg를 사용하시면 좀 힘들거에요..

### Wandb Visualization
This template supports Wandb visualization by using [Wandb](https://github.com/wandb/wandb) library.

#### Quickstart

Get started with W&B in four steps:

1. First, sign up for a [free W&B account](https://wandb.ai/login?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=quickstart).

2. Second, install the W&B SDK with [pip](https://pip.pypa.io/en/stable/). Navigate to your terminal and type the following command:

```bash
pip install wandb
```

3. Third, log into W&B:

```bash
wandb init
```

4. Setting WANDB_NAME of Shell Script and enjoy
```train.sh
WANDB_NAME="YourExperimentName"
```



That's it! Navigate to the W&B App to view a dashboard of your first W&B Experiment. Use the W&B App to compare multiple experiments in a unified place, dive into the results of a single run, and much more!

<p align='center'>
<img src="https://github.com/wandb/wandb/blob/main/docs/README_images/wandb_demo_experiments.gif?raw=true" width="100%">
</p>
<p align = "center">
Example W&B Dashboard that shows Runs from an Experiment.
</p>

&nbsp;

