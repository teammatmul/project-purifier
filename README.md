Project-Purifier
=============================

Project-Purifier는 BERT 모델을 활용한 욕설 판단 및 마스킹 서비스입니다.
library에는 웹사이트 크롤링 코드(Youtube, Naver news, ilbe, namuwiki), 크롤링 데이터 전처리 코드, 한글 구어체 300만 문장이 추가 학습된 pre-trained model weights, 입력 문장 욕설 판단 및 마스킹 코드가 포함되어 있습니다.

저희의 서비스는 google mutilingual BERT model과 HuggingFace의 pytorch BERT 구현을 기반으로 이루어졌으며, puri attention layer의 추가와 몇가지 트릭으로 구성되었습니다. Pytorch 1.~와 Python 3.6 버전에서 테스트 되었습니다.

(We learned how to use pytorch by HuggingFace's code and we made purifier model based on their's. So, special thanks to HuggingFace!)


## 목차
```
1. Project process
2. puri attention
3. 마스킹 알고리즘
4. 코드 사용법
```

## 1. project process

#### 1.1. 데이터 크롤링
- Google BERT multilingual 모델 사용. <정확한 모델 체크>
- 추가 pre-training을 위한 300만 구어체 문장 수집
- fine-tunning을 위한 10만개의 댓글 수집.
 
#### 1.2. 데이터 전처리 및 라벨링
- 구어체 문장 tokenize를 위한 vocab txt 수정 (기존 multilingual 버전으로는 토크나이징 불가)
- 이모티콘 제거 및 데이터 전처리 <뭐했었는지 확인해서 조금 보완>
- fine-tunning 위한 10만개 댓글 욕설유무 0,1 라벨링 <라벨링 기준 표 좀 수정해서 올리기>
 
#### 1.3. 학습 및 모델 선정
- 기존 multilingual 모델에 300만 구어체 문장 추가 학습.
- (구글 BERT tensorflow 코드 사용) <colab 코드 정리 필요>
- 기 학습된 pretrained 모델별 fine-tuning
- 학습 완료된 모델 별 정확도 비교 후 모델 선정 <정확도 비교하는거 보여줄 수 있으면 좋을듯>
- (욕설 판단만 가능한 상태)
 
#### 1.4. 추가 모델링 설계 및 마스킹 알고리즘 설계
- tensorflow -> pytorch로 전환(pre-trained model 또한 변환)
- 형태소별 욕설 여부 라벨링 없이 욕설 위치 파악을 위한 puri attention layer 추가
- Token 단위 확률 비교를 통한 마스킹 알고리즘 구현
- 인자 변경 및 비교를 통해 최선의 모델 선정
 
#### 1.5. 웹 서비스 구현
- AWS EC2(t2.medium, ubuntu), Nginx를 사용하여 서버 구축
- Flask를 사용하여 모델 구동
 
## 2. puri attention
 - 실제 코드에서는 단어 단위로 tokenize 되지는 않으나, 편의상 단어 토큰이라 표현하였습니다.

#### 2.1. 메인 아이디어
- process 3단계까지 완성된 저희의 모델은 입력된 문장 내에 욕설이 있는지 없는지에 대한 classification만 가능했습니다. 물론 이 자체만으로도 문맥적인 욕설을 잡아 낼 수 있다는점에서 기존 rule-based 모델보다 우세하다 할 수 있었습니다만, 문장 내 모든 단어를 순차적으로 조합하는 캐스캐이드 방식으로 마스킹 알고리즘 구현시 2^n번의 예측이 필요하여, 욕설이 있는 부분을 찾아 마스킹하는데에는 부적합 했습니다.
- 하여 모델이 classification을 할 때 어떤 위치 혹은 정보를 기반으로 판단을 하는지를 알아내고, 파악된 위치를 마스킹 하는 방식을 생각하게 되었습니다.

#### 2.2. Attention과 CLS 토큰의 의미
- Attention 함수는 주어진 Q(Query)에 대해서 모든 K(Key)와의 유사도를 구하고, 이 유사도를 가중치로 하여 각각의 V(Value)에 반영해줍니다. BERT는 Q,K,V가 모두 동일한 self attention을 사용하고 있으므로, 이는 입력 문장의 모든 단어 벡터들이 서로를 바라보고 서로를 반영한다 라고 말할수 있습니다.
- BERT는 classification(마지막 softmax layer)에 오직 pooler를 통과한 CLS 토큰 만을 사용합니다. 이는 임베딩 완료된 문장이 12개의 attention layer를 통과하는 동안 문장의 앞뒤 문맥에 대한정보가 CLS 토큰에 담기게 되기 때문입니다. <톺아보기 그림 추가>
- 버트 톺아보기 페이지를 참조한 PCA visualization입니다. 각각의 attention layer output을 투영하여 시각화한 것인데, -12 layer에서는 각각의 벡터들이 한눈에 구분이 가지만, attention layer를 하나씩 통과할수록 점점 섞여감을 알 수 있습니다.
- 즉, classification layer에서 0or1의 판단을 내리게 되는 기준은 CLS 토큰이 되고, CLS 토큰은 전체 문장에 대한 문맥 정보를 갖고 있습니다.(정확히 하자면 모든 토큰이 CLS 토큰처럼 문맥정보가 섞이게 되지만 CLS 토큰만 사용하는게 맞습니다.)

#### 2.3. puri attention
- puri attention layer의 첫번째 핵심은 모든 문맥 정보를 담고있는 CLS 토큰으로 아직 문맥 정보들이 뒤섞이지 않은 임베딩 output을 바라보게(유사도를 구하게) 하는데에 있습니다.(즉, Q는 CLS토큰, K와 V는 임베딩 output이 됩니다)
    
    ```
    CLS token matmul embedding output = AS, AS(softmax)=AP <코드로>
    ```
- 여기서 나온 각 단어 토큰 별 AP를 비교하여(CLS와 SEP 토큰은 mask 처리되어 0, 실제 토큰만 확률을 갖습니다) 일정 이상의 확률일 경우 욕설로 판단하고 해당 단어를 마스킹하게 됩니다.<AP 그래프 사진>
- puri attention layer의 두번째 핵심은 Attention시 Q,K,V의 hidden_state를 없애고 단순 matmul연산으로 바꾸는데에 있습니다. 이는 문맥 정보는 모두 CLS 토큰이 가지고 있으니, 마스킹에 쓰일AP를 계산할때, 각 단어들의 본래 벡터(임베딩된)에 최대한 집중하게 만들기 위해서(weight에 의한 변형 없이)입니다. 또한, classification layer에 들어갈 AP를 최대한 그대로 넣어주고자 함입니다.
- 정리하자면 **"purifier 모델은 puri attention을 통해 fine-tunning 동안 CLS 토큰과 임베딩 처리된 입력 문장의 유사도를 계산하여 그중 값이 높은 토큰을 욕설로 학습해 나간다"** 라고 할 수 있습니다.


## 3. 마스킹 알고리즘

- 문장 전체를 봤을때 욕설 판단이 1로 나오는 경우(욕설이 있는 문장)에만 마스킹 알고리즘 동작.
- 핵심은 puri attention에서 나오는 Attention Prob을 비교하여 높은 값을 욕설이라 판단하는것.
   
- BERT의 tokenize 방식이 단어 혹은 형태소 단위가 아니라, wordpiece 방식으로 구성되어 있어 한 토큰이 일정 확률을 넘어선 경우, 그 토큰을 포함하고 있는 단어 전체를 마스킹 하는 방식으로 구현
   
- 욕이라 판단하는 확률은 1/토큰갯수
   
- <"안녕하세요 씨발"을 예시로 ppt 하나 찍어서 캡쳐해서 올리기>
 
## 4. 코드 사용법

    <puri 모듈화 시켜놓은거 걍 python import 해서 쓰는식으로 몇줄 적기>
