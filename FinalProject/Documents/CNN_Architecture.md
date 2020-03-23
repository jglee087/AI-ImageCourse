## 0. CNN





IVSRC~ 2010년부터 대회 이미지



https://arxiv.org/abs/1901.06032



## 1. LeNet

LeNet은 얀 르쿤 연구팀이 1998년에 개발한 CNN 알고리즘

LeNet-5의 구조

![](../Image/LeNet5.png)



Input층에 입력할 이미지의 크기는 32x32이고, 그 후로 3개의 컨볼루션 층과, 2개의 서브샘플링 층과 1개의 완전 연결층과 output 레이어로 만들어져 있다. 그리고 각 층의 활성화 함수는 모두 tanh이다.



| 레이어        | 특징                                 |
| ------------- | ------------------------------------ |
| C1 레이어     | 5x5 필터가 6개 존재                  |
| S2 레이어     | 2x2 필터를 stride 2로 하여 평균 풀링 |
| C3 레이어     | 14x14 필터가 6개 존재                |
| S4 레이어     | 2x2 필터를 stride 2로 하여 평균 풀링 |
| C5 레이어     | 5x5x16 필터와 컨볼루션               |
| F6 레이어     | 84개의 노드를 가진 신경망            |
| Output 레이어 | 10개의 RBF 유닛들로 구성             |

## 1. LeNet

## 2. AlexNet

## 3. ZfNet

## 4. VGG

## 5. GoogLeNet

## 6. ResNet

## 7. DenseNet

## 8. Xception

## 9. Se-Network





[기타]

WideResNet, Pyramidal Net, ResNeXt



<참고>

https://bskyvision.com/

https://blog.naver.com/laonple/220643128255