## Deal Imbalance Category 



#### Focal Loss

Focal loss는 물체 탐지에 사용하는 **retinanet에서 쓰인 loss function**이다.
목적은 잘 찾은 class 라벨(label)에 대해 Loss를 적게 주어 loss값의 갱신을 거의 하지 못하게 하는 반면에 잘 찾지 못할 것 같은 class label에 대해 loss를 크게 주어 loss 갱신을 크게 하는 것이다. 즉, 데이터가 불균형하게 존재할 때, 상대적으로 class가 적을 때 적은 class를 찾기 위한 방법 중 하나이다. 잘 찾지 못한 class에 대해 더 집중해서 학습하도록 하는 것이 Focal loss 이다.



#### Compute Class Weight



#### Stacking ensemble

[https://inspiringpeople.github.io/data%20analysis/Ensemble_Stacking/](https://inspiringpeople.github.io/data analysis/Ensemble_Stacking/)



#### kullback_leibler_divergence