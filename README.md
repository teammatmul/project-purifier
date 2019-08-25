Project-Purifier
=============================
Project-Purifier는 BERT 모델을 활용한 욕설 판단 및 마스킹 서비스입니다.
library에는 웹사이트 크롤링 코드(Youtube, Naver news, ilbe, namuwiki), 크롤링 데이터 전처리 코드, 한글 구어체 300만 문장이 추가 pre-train 학습 코드, 욕설 판단을 위한 fine-tunning 학습 코드, 입력 문장 욕설 판단 및 마스킹 코드가 포함되어 있습니다.

Purifier 서비스는 [google mutilingual BERT model](https://github.com/google-research/bert)과 [HuggingFace의 pytorch BERT 구현](https://github.com/huggingface/pytorch-transformers)을 기반으로 이루어졌으며, puri attention layer의 추가와 몇가지 트릭으로 구성되었습니다. torch1.1과 Python 3.6 버전에서 테스트 되었습니다.

학습 관련 디바이스는 모두 [colab](https://colab.research.google.com)의 T4 gpu를 사용했고, 관련 notebook 파일도 정리해 두었습니다.

(We learned how to use pytorch by HuggingFace's code and we made purifier model based on their's. So, special thanks to HuggingFace!)


## 목차

> ### 1. Project process
> ### 2. Puri attention
> ### 3. 마스킹 알고리즘
> ### 4. 코드 사용법


## 1. Project process

#### 1.1. 데이터 크롤링
- Google BERT-base multilingual pre-trained 모델 사용.
- 추가 pre-training을 위한 300만 구어체 문장 수집
- fine-tunning을 위한 10만개의 댓글 수집.
 
#### 1.2. 데이터 전처리 및 라벨링
- 구어체 문장 tokenize를 위한 vocab.txt 수정 (기존 multilingual 버전으로는 토크나이징 불가)
- 이모티콘 제거 및 데이터 전처리
- fine-tunning 위한 10만개 댓글 욕설유무 0,1 라벨링
- 욕설의 기준(라벨링 방향성)
> 비속어 욕설은 무조건 1,
> 부모 관련 욕설(aka 패드립), 충분히 순화가 가능하나 남을 비하하려는 의도의 과격한 표현,
> 지역 갈등을 조장하는 표현, 부정적인 프레임을 씌우는 용어를 욕설로 판단.
 
#### 1.3. 욕설 학습 및 모델 선정
- 기존 multilingual 모델에 300만 구어체 문장 추가 학습.
- (구글 BERT tensorflow 코드 사용)
- 기 학습된 pretrained 모델별 fine-tuning
- 학습 완료된 모델 별 정확도 비교 후 모델 선정 (욕설 판단만 가능한 상태)
 
#### 1.4. 마스킹을 위한 추가 모델링 설계 및 마스킹 알고리즘 설계
- tensorflow -> pytorch로 전환(pre-trained model 또한 변환)
> 코드 추가 및 변경의 편리함, bertviz를 사용하기 위함
- 형태소별 욕설 여부 라벨링 없이 욕설 위치 파악을 위한 puri attention layer 추가
- Token 단위 확률 비교를 통한 마스킹 알고리즘 구현
- 인자 변경 및 비교를 통해 최선의 모델 선정
 
#### 1.5. 웹 서비스 구현
- AWS EC2(t2.medium, ubuntu) 기반
- Nginx, uwsgi 사용하여 웹 서버 구축
- Flask와 Redis를 사용하여 모델 서버 구축
 
## 2. Puri attention
 - 실제 코드에서는 단어 단위로 tokenize 되지는 않으나, 편의상 단어 토큰이라 표현하였습니다.

#### 2.1. 메인 아이디어
- process 3단계까지 진행된 purifier 모델은 입력된 문장 내에 욕설이 있는지 없는지에 대한 classification만 가능했습니다. 물론 이 자체만으로도 문맥적인 욕설을 잡아 낼 수 있다는점에서 기존 rule-based 모델보다 우세하다 할 수 있지만, 문장 내 모든 단어를 순차적으로 조합하는 캐스캐이드 방식으로 마스킹 알고리즘을 구현할 경우 2^n번의 예측이 필요하여, 욕설이 있는 부분을 찾아 마스킹하는데에는 부적합 했습니다.
- 하여 모델이 classification을 할 때 어떤 위치 혹은 정보를 기반으로 판단을 하는지를 알아내고, 파악된 위치를 마스킹 하는 방식을 생각하게 되었습니다.

#### 2.2. Attention과 CLS 토큰의 의미
- Attention 함수는 주어진 Q(Query)에 대해서 모든 K(Key)와의 유사도를 구하고, 이 유사도를 가중치로 하여 각각의 V(Value)에 반영해줍니다. BERT는 Q,K,V가 모두 동일한 self attention을 사용하고 있으므로, 이는 **입력 문장의 모든 단어 벡터들이 서로를 바라보고 서로를 반영한다** 라고 말할수 있습니다.
- BERT는 classification(마지막 softmax layer)에 pooler를 통과한 CLS 토큰 만을 사용합니다. 이는 임베딩 완료된 문장이 12개의 attention layer를 통과하는 동안 문장의 앞뒤 문맥에 대한정보가 CLS 토큰에 담기게 되기 때문입니다.

![톺아보기](/img/pool_mean.png)

- [버트 톺아보기 페이지](http://docs.likejazz.com/bert/)를 참조한 PCA visualization입니다. 각각의 attention layer output을 투영하여 시각화한 것인데, -12 layer에서는 각각의 벡터들이 한눈에 구분이 가지만, attention layer를 하나씩 통과할수록 점점 섞여가고 있는 것을 알 수 있습니다.
- 즉, classification layer에서 0or1의 판단을 내리게 되는 기준은 CLS 토큰이 되고, CLS 토큰은 전체 문장에 대한 문맥 정보를 갖고 있습니다.(정확히 하자면 모든 토큰이 CLS 토큰처럼 문맥정보가 섞이게 되지만 CLS 토큰만 사용하는게 맞습니다.)

#### 2.3. puri attention
- puri attention layer의 첫번째 핵심은 모든 문맥 정보를 담고있는 CLS 토큰으로 아직 문맥 정보들이 뒤섞이지 않은 임베딩 output을 바라보게(유사도를 구하게) 하는데에 있습니다.(즉, Q는 CLS토큰, K와 V는 임베딩 output이 됩니다)
    
    ```
    [CLS token] matmul [embedding output] = Attention Score
    Attention Prob = softmax(Attention Score)
    ```
- 여기서 나온 각 단어 토큰 별 Attention Prob(AP)을 비교하여(CLS와 SEP 토큰은 mask 처리로 인해 그 값이 0이 되고, 실제 토큰만 확률을 갖습니다) 일정 이상의 확률일 경우 욕설로 판단하고 해당 단어를 마스킹하게 됩니다.

     예문: '씨발새1끼님아 제에발 잘좀 해주셨음 좋겠어요. 아시겠어요 병신아?'
![ap_graph](/img/ap_graph.png)


- puri attention layer의 두번째 핵심은 Attention시 V의 hidden_state를 없애고 단순 matmul연산으로 바꾸는데에 있습니다. 이는 문맥 정보는 모두 CLS 토큰이 가지고 있으니, 마스킹에 쓰일 AP를 계산할때, 각 단어들의 본래 벡터(임베딩된)에 최대한 집중하게 만들기 위해서(weight에 의한 변형 없이)입니다. 또한, classification layer에 들어갈 AP를 최대한 그대로 위해서 입니다.

- 정리하자면 **"purifier 모델은 puri attention을 통해 fine-tunning 동안 CLS 토큰과 임베딩 처리된 입력 문장의 유사도를 계산하여 그중 값이 높은 토큰을 욕설로 학습해 나간다"** 라고 할 수 있습니다.

#### 2.4. Experiment
- 기존 attention layer에서 CLS 토큰이 자기 자신을 바라보는 경우에도 attention mask를 씌워보았으나 큰 차이는 없었습니다.
- Puri attention layer에서 Q,K,V 값을 인자로 설정할 수 있게 하여 여러가지 경우의 수를 실험하였습니다.
- Q (CLS 토큰) 의 경우 12번째 attention layer를 통과한 후 기존 pooler를 통과한 경우가 조금 더 높은 정확도를 얻을 수 있었습니다.
- K,V 의 경우 꼭 Embedding output이 아니라 attention layer의 초반부(1~3) output 을 조합하여 Layer Normalization을 적용해 사용한 경우에 조금 더 높은 정확도를 얻을 수 있었습니다.
- Q,K의 hidden_state를 없애주는 경우에도 유사한 결과를 얻을 수 있었으나, 둘 다 없애는 경우에는 문장 내 욕설 유무 판단에서 현저히 낮은 정확도를 가져왔습니다. 하지만 AP의 욕설과 비욕설 단어의 확률 차이를 내는데에는 없애주는 경우가 더 유용했습니다.
- 최종 모델은 문장 내 욕설의 유무 판단과 욕설의 위치를 찾는 성능을 종합하여 선정하였습니다.

## 3. 마스킹 알고리즘

- 문장 전체를 봤을때 욕설 판단이 1로 나오는 경우(욕설이 있는 문장)에만 마스킹 알고리즘이 적용됩니다.
- 핵심은 puri attention에서 나오는 Attention Prob을 비교하여 가장 높은 값을 욕설이라 판단하는 것입니다.
   
- BERT의 tokenize 방식이 단어 혹은 형태소 단위가 아니라, wordpiece 방식으로 구성되어 있어 한 토큰이 일정 확률을 넘어선 경우, 그 토큰을 포함하고 있는 단어 전체를 마스킹 하는 방식으로 구현하였습니다.
   
- 가장 높은 확률의 토큰이 포함된 단어를 마스킹하고, 욕설 판단이 0(욕설이 없는 문장)이 될때까지 같은 과정을 반복합니다.
   
- <"안녕하세요 씨발 반가워요 개돼지!"를 예시로 순서대로 바뀌는거 찍어서 캡쳐해서 올리기>
 
## 4. 코드 간단 사용법
    ```
    import purifier_cls as puri
    from final_modeling import BertForSequenceClassification
    
    model = BertForSequenceClassification.from_pretrained('./data/', num_labels=2)
    puri.single_sentence_masking_percent("안녕하세요 씨발 반가워요 개돼지!", model)
    ```
