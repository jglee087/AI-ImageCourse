## Radam



케라스를 사용하는 방법은 다음과 같다.

1. pip install keras-rectified-adam 로 인스톨 한다.

2. from keras_radam import RAdam 로 불러온다.

3. model.compile(RAdam()) 이런식으로 컴파일 할때 옵티마이저 부분을 대체해 줍니다.

4.  RAdam 의 옵션은 Adam과 동일하고 그리고 total_steps=5000, warmup_proportion=0.1, min_lr=1e-5 다음과 같이 덧붙여 줄 수 있다.

<참고>

https://github.com/CyberZHG/keras-radam



https://theonly1.tistory.com/1754
https://github.com/LiyuanLucasLiu/RAdam
https://blog.naver.com/PostView.nhn?blogId=horajjan&logNo=221789504157&parentCategoryNo=&categoryNo=118&viewDate=&isShowPopularPosts=true&from=search

https://medium.com/vitalify-asia/whats-up-with-deep-learning-optimizers-since-adam-5c1d862b9db0

https://www.pyimagesearch.com/2019/10/07/is-rectified-adam-actually-better-than-adam/



Adaptive learning rate를 사용하는 optimizer들은 수렴 속도가 빠르지만 초기 학습때 gradient 분포가 왜곡되어 local optima에 빠지는 단점이 있다.
그래서 초기 learning rate를 매우 작게 설정하는 warm up 방법을 사용해서 이 단점을 극복한다.

