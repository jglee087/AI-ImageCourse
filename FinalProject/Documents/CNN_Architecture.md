## 0. CNN



이미지 분류 모델 평가에 사용되는 top-5 error와 top-1 error



## 0. **ILSVRC**

ILSVRC은 ImageNet Large Scale Visual Recognition Challenge의 약자로 이미지 인식 경진대회이다. 
이미지 인식과 이미지 분류는 같은 의미
대용량의 이미지셋을 주고 이미지 인식 알고리즘의 성능을 평가하는 대회



2012년 이후로는 CNN 기반 딥러닝 알고리즘 AlexNet이 우승을 차지한 이후에는 깊은 구조를 가진 알고리즘들이 우승을 차지

![](../Image/ILSVRC.png)



## 1. LeNet

LeNet은 얀 르쿤 연구팀이 1998년에 개발한 CNN 알고리즘

LeNet-5의 구조

![](../Image/LeNet5.png)



Input층에 입력할 이미지의 크기는 32x32이고, 그 후로 3개의 컨볼루션 층과, 2개의 서브샘플링 층과 1개의 완전 연결층과 output 레이어로 만들어져 있다. 그리고 각 층의 활성화 함수는 모두 tanh이다.



| 레이어        | 특징                                       |
| ------------- | ------------------------------------------ |
| C1 레이어     | 5x5 필터가 6개 존재                        |
| S2 레이어     | 2x2 필터를 stride 2로 하여 평균 풀링       |
| **C3** 레이어 | 5x5 필터가 16개 존재                       |
| S4 레이어     | 2x2 필터를 stride 2로 하여 평균 풀링       |
| C5 레이어     | 5x5x16 필터와 컨볼루션                     |
| F6 레이어     | 84개의 노드를 가진 신경망                  |
| Output 레이어 | 10개의 RBF 유닛들로 구성(Machine Learning) |



C3 레이어에서 특성맵을 조합하는 방법

![](../Image/LeNet5_2.png)

1. 6장의 14 x 14 특성맵에서 연속된 3장씩을 모아서 5 x 5 x 3 사이즈의 필터와 컨볼루션 해준다. 6장의 10 x 10 특성맵이 산출 

2. 6장의 14 x 14 특성맵에서 연속된 4장씩을 모아서 5 x 5 x 4 사이즈의 필터와 컨볼루션 해준다. 6장의 10 x 10 특성맵이 산출

3. 6장의 14 x 14 특성맵에서 불연속한 4장씩을 모아서 5 x 5 x 4 사이즈의 필터와 컨볼루션 해준다. 3장의 10 x 10 특성맵이 산출 

4. 6장의 14 x 14 특성맵 모두를 가지고 5 x 5 x 6 사이즈의 필터와 컨볼루션 해준다. 1장의 10 x 10 특성맵이 산출

총 16장의 10 x 10 특성맵 



## 2. AlexNet

![](../Image/AlexNet.png)

AlexNet은 5개의 컨볼루션 레이어와 3개의 full-connected 레이어로 총 8개의 레이어로 구성

두번째, 네번째, 다섯번째 컨볼루션 레이어들은 전 단계의 같은 채널의 특성맵들과만 연결
**세번째 컨볼루션 레이어**는 전 단계의 두 채널의 특성맵들과 모두 연결

 AlexNet에 입력 되는 이미지의 크기는 224 x 224 x 3



### AlexNet의 특징

1. ReLU 함수

   같은 정확도를 유지하면서 tanh을 사용하는 것보다 6배나 빠르다고 한다.

   ![](../Image/ReLU.png)

   

2. Dropout

   Fully-connected layer의 뉴런 중 일부를 생략하면서 학습을 진행

   ![](../Image/Dropout.png)

   

3. Overlapping Pooling

   LeNet-5의 경우 평균 풀링(average pooling)이 사용, AlexNet에서는 **최대 풀링(max pooling)**이 사용

   LeNet-5는 **non-overlapping 평균 풀링**을 사용, AlexNet은 **overlapping 최대 풀링**을 사용

   

   ![](../Image/OverlappingPooling.png)

4. Local response normalization (LRN)

   ReLU는 양수의 방향으로는 입력의값을 그대로 사용
   그렇게되면 CONV나 POOLING시 매우 높은 하나의 픽셀값이 주변의 픽셀에 영향
   이것을 방지하기 위해 다른 ActivationMap의 같은 위치에있는 픽셀끼리 정규화

   

5. Data augmentation

   - 과적합을 막는 가장 좋은 방법 중 하나는 데이터의 양을 늘리는 것으로 과적합을 막기 위해 사용

   - 훈련시킬 때 적은 양의 데이터를 가지고 훈련시킬 경우 과적합될 가능성이 큼

   

   

   ![](../Image/DataAug.png)




## 3. ZFNet



![](../Image/ZFnet.png)



AlexNet을 기반으로 첫 Conv layer의 filter size를 11에서 7로, stride를 4에서 2로 바꾸고, 그 뒤의 Conv layer들의 filter 개수를 키워주는 등(Conv3,4,5: 384, 384, 256 –> 512, 1024, 512) 약간의 튜닝을 거침

이 모델은 architecture에 집중하기 보다는, 학습이 진행됨에 따라 feature map을 시각화하는 방법과, 모델이 어느 영역을 보고 예측을 하는지 관찰하기 위한 occlusion 기반의 attribution 기법 등 **시각화 측면**에 집중



![](../Image/ZFnet2.png)





## 4. VGG



2014년 이미지넷 이미지 인식 대회에서 준우승을 한 모델로 VGGNet은 16개(VGG16) 또는 19개(VGG19)의 층으로 구성된 모델을 의미

VGGNet 모델부터 시작해서 네트워크의 깊이가 확 깊어짐

2012년, 2013년 우승 모델들은 8개의 층으로 구성
2014년의 VGGNet(VGG19)는 19층으로 구성
GoogLeNet은 22층으로 구성



![](../Image/VGG16.png)





![](../Image/VGG.png)

 3 x 3 필터가 3개면 총 27개의 가중치를 갖는다. 반면 7 x 7 필터는 49개의 가중치를 갖는다. 

CNN에서 가중치는 모두 훈련이 필요한 것들이므로, 가중치가 적다는 것은 그만큼 훈련시켜야할 것의 갯수가 작아진다. 따라서 학습의 속도가 빨라진다. 동시에 층의 갯수가 늘어나면서 특성에 비선형성을 더 증가시키기 때문에 특성이 점점 더 유용





## 5. GoogLeNet(Inception_v1)



![](../Image/GoogLeNet.png)











#### (1) **1 x 1 컨볼루션**

![](../Image/1x1Conv.png)



![](../Image/1x1Conv2.png)



Bottleneck Structure

dimension reduction



#### (2) **Inception 모듈**



![](../Image/Inception2.png)



![](../Image/Inception1.png)



#### (3) global average pooling





![](../Image/GlobalPooling.png)



#### **(4) auxiliary classifier**



![](../Image/AuxClassifier.png)



- softmax영역이 망 전체에 총 3 개가 있다.
- 실제 마지막 Layer 가 진짜배기 *softmax* 레이어이고 나머지 2개는 보조 *softmax* 레이어
- 이는 망이 깊어질 때 발생하는 vanishing 문제를 해결하고자 중간 층에서도 backpropagtion을 수행하여 weight 갱신을 시도

https://norman3.github.io/papers/docs/google_inception.html

https://datascienceschool.net/notebook/ETC/



## 5-1. Inception. v2

더 작은 단위의 conv 를 사용

- 예제로 보는 Conv Factorization.



1. 비대칭(Asymmetric) Conv 를 사용한 Factorization



2. 보조 분류기 (Auxiliary Classifiers)

- 이 중에 맨 하단 분류기는 삭제한다. 실험을 통해 성능에 영향을 주지 못하는 것으로 확인되었다.



3. 효율적인 그리드(Grid) 크기 줄이기

    Representation Bottleneck 을 없애는 구조

   ![]()



![](../Image/Inception_v2.png)





![](../Image/Inception_v2_str.png)



## 5-2. Inception_v3



Inception.v3는 Inception.v2 를 만들고 나서 이를 이용해 이것 저것 수정해보다가 결과가 더 좋은 것들을 묶음

- **RMSProp** : Optimizer를 바꾼거다.
- Label Smoothing
  - 논문에 자세히 나와있긴 한데 간단히 설명하자면 Target 값을 one-hot encoding을 사용하는 것이 아니라,
  - 값이 0 인 레이블에 대해서도 아주 작은 값 ee 를 배분하고 정답은 대충 1−(n−1)∗e1−(n−1)∗e 로 값을 반영하는 것이다.
- Factorized 7-7
  - 이게 좀 아리까리한게 맨 앞단 conv 7x7 레이어를 (3x3)-(3x3) 2 레이어로 Factorization 한 것이라고 한다. (앞에서 설명한 것이다.)
  - 그런데 v2 레이어 표를 보면 이미 적용되어 있는 것 같기도 해서 혼동이…
  - 어쨋거나 이를 적용했다고 한다. 일단 넘어가자.
- BN-auxiliary
  - 마지막 Fully Conntected 레이어에 Batch Normalization(BN)을 적용한다.



## 5-3. Inception_v4



Inception v4 model structure

![](../Image/Inception_v4.png)

https://norman3.github.io/papers/docs/google_inception.html





## 6. ResNet



![](../Image/DeepLayer.png)





![](../Image/Residual.png)







![](../Image/ResNet.png)





![](../Image/ResNet_2.png)







## 7. DenseNet



![](../Image/DenseNet.png)

장점

1. Vanishing-gradient 문제 완화
2. 더 강력한 피쳐 전달이 가능
3. 피쳐 재사용을 촉진
4. 파라미터의 수를 감소
5. Regularlizing 효과와 Overfitting 감소



![](../Image/DenseNet121.png)



Bottleneck layers





## 8. Xception

Depth-wise separable convolution

 Inception은 연산량을 줄여 더 깊은 모델을 만들고자 했다고 하면 Xception은 이 모델의 prameter들을 더 효율적으로 사용하기위한 방법을 제시



## 9. Se-Network







[기타]

WideResNet, Pyramidal Net, ResNeXt



<참고>

https://arxiv.org/abs/1901.06032

https://bskyvision.com/

https://blog.naver.com/laonple/220643128255

https://j911.me/2019/07/densenet.html

https://datascienceschool.net/view-notebook/4ca30ffdf6c0407ab281284459982a25/

https://hichoe95.tistory.com/49

https://hoya012.github.io/blog/deeplearning-classification-guidebook-2/