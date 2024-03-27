# L7. Classification Problems

- [L7. Classification Problems](#l7-classification-problems)
  - [0. 문제상황](#0-문제상황)
  - [1. 소스코드](#1-소스코드)
  - [2. 사용 Package, Module, Class, Functions 설명](#2-사용-package-module-class-functions-설명)
    - [2.1. torchvision](#21-torchvision)
      - [2.1.1. (CLASS) torchvision.transforms.Compose(transforms)](#211-class-torchvisiontransformscomposetransforms)
      - [2.1.2. (CLASS) torchvision.transforms.ToTensor](#212-class-torchvisiontransformstotensor)
      - [2.1.3. (CLASS) torchvision.transforms.Normalize(mean, std, inplace=False)](#213-class-torchvisiontransformsnormalizemean-std-inplacefalse)
      - [2.1.4. (CLASS) torchvision.datasets.MNIST(root: Union\[str, Path\], train: bool = True, transform: Optional\[Callable\] = None, target\_transform: Optional\[Callable\] = None, download: bool = False)](#214-class-torchvisiondatasetsmnistroot-unionstr-path-train-bool--true-transform-optionalcallable--none-target_transform-optionalcallable--none-download-bool--false)
    - [2.2. torch](#22-torch)
      - [2.2.1. (CLASS) torch.utils.data.DataLoader(dataset, batch\_size=1, shuffle=None, sampler=None, batch\_sampler=None, num\_workers=0, collate\_fn=None, pin\_memory=False, drop\_last=False, timeout=0, worker\_init\_fn=None, multiprocessing\_context=None, generator=None, \*, prefetch\_factor=None, persistent\_workers=False, pin\_memory\_device='')](#221-class-torchutilsdatadataloaderdataset-batch_size1-shufflenone-samplernone-batch_samplernone-num_workers0-collate_fnnone-pin_memoryfalse-drop_lastfalse-timeout0-worker_init_fnnone-multiprocessing_contextnone-generatornone--prefetch_factornone-persistent_workersfalse-pin_memory_device)

## 0. 문제상황

본인이 입사한 ㈜OO 기업은 외부로부터 문제를 의뢰받아 딥러닝 기반 솔루션을 개발 및 납품하는 기업이다.   
본인이 입사 후 담당하게 된 첫 업무는 정부기관으로부터 의뢰받은 `수기 숫자 인식(handwritten digit recognition)` 문제이다.   

해당 정부 기관은 다양한 종이 양식에 수기로 작성된 숫자를 인식하여 데이터베이스화하는 작업을 자동화하려고 있다. 이를 위해 필요한 것은 문서 속에서 발견되는 `각각의 수기 숫자 문자`를 `디지털 심볼`로 변환하도록 `인공신경망`을 학습하는 일이다.   

이를 위해서는 `MNIST 데이터베이스` (Modified National Institue of Standards and Technology database)를 활용
할 것을 요청받았다. 해당 데이터는 미국의 정부 공무원과 고등학생들이 작성한 내용을 바탕으로 수집된 데이터이고, `10가지 숫자`에 대한 `6만개의 표본`을 포함하고 있다. 본인은 몇 주 안에 개발을 완료해야 한다…

## 1. 소스코드
[pbl_pro_1.py](./pbl_pro_1.py)

## 2. 사용 Package, Module, Class, Functions 설명
### 2.1. torchvision
`이미지, 비디오 데이터 처리` 및 `컴퓨터 비전 작업`을 위한 패키지이다.
#### 2.1.1. (CLASS) torchvision.transforms.Compose(transforms)
- Intro:  
  여러 개의 변환을 조합하여 하나의 `전처리 파이프라인`을 만든다.   
  이 파이프라인은 데이터셋에서 `이미지`를 가져와서 `모델에 입력으로 사용할 수 있는 형식`으로 변환한다.  

- Parameters:  
  |Param|Type|Desc|
  |:--:|:--:|:--:|
  |`transforms`|list of Transform objects|transforms objects의 리스트|

- [문서 바로가기](https://pytorch.org/vision/main/generated/torchvision.transforms.Compose.html)

#### 2.1.2. (CLASS) torchvision.transforms.ToTensor
- Intro:  
  이미지를 `Pytorch텐서`로 변환한다. 이미지 데이터의 픽셀 값 범위를 `[0,255]` 에서 `[0.0, 1.0]`으로 `정규화`한다.

- [문서 바로가기](https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html)

#### 2.1.3. (CLASS) torchvision.transforms.Normalize(mean, std, inplace=False)
- Intro:  
  `정규화`를 수행한다. 입력 이미지의 각 채널에 대해 `평균을 빼고` `표준편차로 나누어` 픽셀 값을 정규화한다.  
  모델의 학습을 `안정화`시키고, `성능을 향상`시키는 `전처리 기법`이다.

- Parameters:  
  |Param|Type|Desc|
  |:--:|:--:|:--:|
  |`mean`|sequence|각 채널의 `평균값`|
  |`std`|sequence|각 채녈의 `표준편차`|
  |`inplace`|bool,optional|입력 데이터를 수정할 것인지 <br>OR<br> 새로운 텐서를 생성할 것인지|

- [문서 바로가기](https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html)

#### 2.1.4. (CLASS) torchvision.datasets.MNIST(root: Union[str, Path], train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
- Intro:  
  `MNIST` 데이터셋을 로드하는 PyTorch의 함수이다.  
  `MNIST`는 `손으로 쓴 숫자 이미지` 데이터셋으로, 컴퓨터 비전 분야에서 많이 사용되는 벤치마크 데이터셋 중 하나이다.

- Parameters:  
  |Param|Type|Desc|
  |:--:|:--:|:--:|
  |`root`|Union[str, Path]|데이터셋이 저장될 `루트 디렉토리의 경로`를 지정합니다. <BR> 데이터셋이 이 디렉토리에 다운로드되거나 저장됩니다.|
  |`train`|bool, optional|데이터셋이 `훈련 데이터`인지 `테스트 데이터`인지를 지정합니다.<BR> 기본값은 `True`로, 훈련 데이터셋을 로드합니다.|
  |`transform`|Optional[Callable], optional|이미지 데이터에 적용할 `변환(transform)`을 지정합니다. <BR> 이 매개변수를 통해 이미지 데이터를 `전처리`하거나 변형할 수 있습니다. <BR> 기본값은 `None`으로, 변환을 적용하지 않습니다.|
  |`target_transform`|Optional[Callable], optional|타겟(레이블) 데이터에 `적용할 변환`을 지정합니다. <BR>예를 들어, 타겟 데이터에 `원-핫 인코딩`을 적용하거나 <BR>특정 형식으로 변환하는 등의 작업을 수행할 수 있습니다. <BR>기본값은 `None`으로, 변환을 적용하지 않습니다.|
  |`download`|bool, optional|데이터셋을 `다운로드할지 여부`를 지정합니다. <BR>만약 데이터셋이 로컬에 존재하지 않으면 `True`로 설정하면 `자동으로 다운로드`됩니다. <BR>기본값은 `False`로, 다운로드를 수행하지 않습니다.|

- [문서 바로가기](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html)

### 2.2. torch
#### 2.2.1. (CLASS) torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None, *, prefetch_factor=None, persistent_workers=False, pin_memory_device='')
- Intro:  
    `PyTorch`에서 데이터를 `미니배치(mini-batch)`로 분할하고 데이터를 로드하는 역할을 한다.  
    주로 데이터셋을 반복하면서 모델에 공급할 수 있는 형식으로 데이터를 로드할 때 사용된다. 

- Parameters:  
  |Param|Type|Desc|
  |:--:|:--:|:--:|
  |`dataset`|Dataset|DataLoader에 제공될 `데이터셋`입니다. <br>PyTorch의 `torch.utils.data.Dataset` 클래스를 상속한<br> 사용자 정의 데이터셋 객체를 사용합니다.|
  |`batch_size`|int, optional|`미니배치`의 크기를 지정합니다. <br>DataLoader는 주어진 배치 크기에 따라 데이터를 미니배치로 나누어 제공합니다.|
  |`shuffle`|bool, optional|`데이터를 섞을지 여부`를 지정합니다. <br>True로 설정하면 데이터가 에폭마다 섞입니다.|
  |`생략`|***|***|

- [문서 바로가기](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)