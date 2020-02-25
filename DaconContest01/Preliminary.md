Image classification은 2012년 AlexNet이 발표되면서 CNN 기술을 이용하여 기존 컴퓨터 비전 기술 대비 압도적인 성능차를 보여줬기 때문에 classification에 CNN을 적용하는 것은 너무 당연하게 인식되어 왔고 현재는classification 분야에서는 이미 사람의 수준을 넘어섰다고 알려져있다.

이미지나 영상의 중요한 응용분야 중 하나인 이미지나 영상에서 의미 있는 부분으로 구별을 해내는 이미지 분할(image segmentation)이다. 이 이미지 분할은 Semantic segmentation은 이미지 내에서 픽셀 단위로 객체를 분류해내는 작업을 의미한다. 분류와 같이 특정 기준점이나 가장자리에 기반한 단순 구별이 아니라, 영상에서 의미 있는 부분으로 구별해내는 기술을 Semantic segmentation이라고 하며, detection 분야가 CNN을 이용하여 성능이 많이 개선되었듯이 semantic segmentation 분야 역시 CNN 기술을 활용하여 상당한 성과를 거두었다.



분류(classification)는 전체 영상에서 특정 객체가 무엇인지 단지 파악만 하는 것이라면 detection은 위치를 파악하고 해당 객체를 감싸는 bounding box로 구별까지 해주어야 하기 고급 기술이 필요하다. 반면에 분할(segmentation)은 bounding box로 대강의 영역을 표시하는 것이 아니라 개체의 정확한 경계선까지 추출해주어야 한다.



#### Image Segmentation

Segmentation의 목표는 영상을 의미적인 면이나 인지적인 관점에서 서로 비슷한 영역으로 영상을 분할하는 것을 것이다. 이미지나 영상에서 특정 물체의 경계면을 정확히 검출하는 것이 좋겠지만 어려운 기술이다.  Segmentation은 영상을 분석하고 이해하는데 필요한 기본적인 단계이며, 영상을 구성하고 있는 주요 성분으로 분해를 하고, 관심인 객체의 위치와 외곽선을 추출해내는 것이다.

Segmentation의 접근 방법에 따라 크게 3가지 방식이 있다.

- 픽셀 기반 방법: 이 방법은 흔히 thresholding에 기반한 방식으로 histogram을 이용해 픽셀들의 분포를 확인한 후 적절한 threshold를 설정하고, 픽셀 단위 연산을 통해 픽셀 별로 나누는 방식이며, 이진화에 많이 사용이 된다. thresholding으로는 전역(global) 혹은 지역(local)로 적용하는 영역에 따른 구분도 가능하고, 적응적(adaptive) 혹은 고정(fixed) 방식으로 경계값을 설정하는 방식에 따른 구별도 가능하다.
- 가장자리 기반 방법:  Edge를 추출하는 필터 등을 사용하여 영상으로부터 경계를 추출하고, 흔히 non-<u>maximum suppression</u>과 같은 방식을 사용하여 의미 있는 edge와 없는 edge를 구별하는 방식을 사용한다.
- 영역 기반 방법: Thresholding이나 Edge에 기반한 방식으로는 의미 있는 영역으로 구별하는 것이 쉽지 않으며, 특히 잡음이 있는 환경에서 결과가 좋지 못하다. 하지만 영역 기반의 방법은 기본적으로 영역의 동질성(homogeneity)에 기반하고 있기 때문에 다른 방법보다 의미 있는 영역으로 나누는데 적합하지만 <u>동질성을 규정하는 방법</u>을 어떻게 정할 것인가가 관건이다. 흔히 seed라고 부르는 몇 개의 픽셀에서 시작하여 영역을 넓혀가는 region growing 방식이 여기에 해당이 된다. 이외에도 region merging, region splitting, split and merge, watershed 방식 등도 있다.

https://blog.naver.com/laonple/220873446440