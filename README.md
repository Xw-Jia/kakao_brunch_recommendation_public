## 사용 방법

1. datasets 폴더에 대회의 데이터셋을 넣는다.
2. python train.py 를 통해 학습한다.
3. python inference.py 를 통해 추론한다.

## 사용한 라이브러리

* torch
* pandas
* numpy
* scipy
* lightgbm
* tqdm

## config.py 파라미터 설명

* `dataset_root` : dataset 이 들어 있는 폴더의 상대 주소
* `models_root` : 여러 모델 관련 파일을 저장할 폴더의 상대 주소
* `result_path` : 결과물을 저장할 파일 경로
* `save_memory` : True로 할 시 모델의 전체 크기가 1기가를 넘지 않지만 매우 느릴 수 있음.
* `heavy_user_threshold` : 이 숫자 이상의 글을 읽었을 경우 헤비 유저로 간주
* `heavy_item_threshold` : 이 숫자 이상의 유저가 읽었을 경우 헤비 아티클로 간주
* `mf_factors` : Matrix Factorization 모델의 latent vector size
* `mf_iterations` : Matrix Factorization 모델의 학습 iteration 수
* `batch_size, num_workers, optimizer, learning_rate, device, device_idx, num_gpu` : obvious
* `reader_count, article_count` : 하드코딩 해놓은 전체 유저/아티클 개수
* `gbm_n_estimators` : lightgbm ranker 가 학습할 estimator의 최대 개수
* `train_sample_size` : 트레이닝 할 시 사용할 유저의 개수
* `val_sample_size` : 검증 시 사용할 유저의 개수
* `submit_user_set` : 제출에 사용할 유저들 ('dev'/'test' 중 하나)
* `candidate_size` : 후보 생성 시 기준이 되는 숫자

## 모델 메모리 관련

### Matrix Factorization

현재 세팅에서 MF 모델은 embedding size 8을 사용하고 있기 때문에 weight 를 저장하면 35MB가 나옵니다.

### Item2Item

현재 세팅에서 Item2Item 모델은 7597x7597 크기의 sparse matrix 를 사용하여 점수를 저장해 둡니다.

저장할 경우 641MB 입니다.

matrix 의 주어진 row에서 score 순으로 정렬된 column 들을 얻어야 하는 operation이 자주 쓰이기 때문에

row 마다 정렬된 column들을 미리 계산해서 저장해 놓은 dict를 사용하면 실행 속도가 훨씬 빠릅니다.

다만 저 dict까지 만들어 버리면 모델의 메모리 상에서의 크기가 1기가가 넘을 것으로 생각됩니다.

config.py 파일에서 save_memory 값을 True로 하면 저 dict를 만들지 않고 매번 정렬을 수행합니다. 그 대신 속도가 매우 느릴 수 있습니다.

save_memory 값을 False로 설정해두면 훨씬 빠르게 실행할 수 있지만 대회의 메모리 제한을 맞추기 위해 공식적인 값은 True로 해두었습니다.

### Ranker

lightgbm 의 LGBMRanker 클래스를 사용해서 랭킹을 수행합니다.

LGBMRanker를 통쨰로 저장할 경우 840KB의 크기를 갖습니다.

---

위에서 명시된 세 가지가 사용하는 모델의 전부입니다.

모델의 핵심적인 부분을 저장한 파일들의 크기의 합이 700MB를 넘지 않기 때문에, 실제 메모리 상에서 세 모델의 크기의 합도 1기가를 넘지 않을 것이라고 생각합니다.
