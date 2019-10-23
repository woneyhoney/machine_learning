# machine_learning

< Assignment 1 - Linear Regression >

1. computeCost.m: 현재 주어진 X값과 정답 y값을 가지고 현재까지 구해진 theta값을 이용해서 최종 cost function 값인 J를 계산하는 함수

2. gradientDescent.m: 현재 주어진 X값과 정답 y값 그리고 현재까지 구해진 theta값을 이용하고 이때 learning rate인 alpha값을 입력하으로 하여

 이후 theta값을 gradient descent 방식으로 계산하여 입력인 num_iters (iteration수)를 이용해서 J값의 변화 history를 vector형식으로 저장하는 함수

3. output.txt (실행후 생성된 결과 파일)


< Assignment 2 - Regularized Logistic Regression >

1. costFunctionReg.m: Logistic Regression with Regularization에 대한 Cost Function

line 23: regCost = (lambda / (2 * m)) * norm(theta([2:end])) ^ 2;
-> 1 - alpha * (lambda / m) = 0.99 로 가정 ( alpha = learning rate, lambda = regularization parameter, m = no. of traning examples) 
-> Gradient Descent의 one iteration에서 theta가 0.99배가 되니 square norm oftheta가 조금 더 작아지는 것을 반영하기 위해 norm(theta([2:end])) ^ 2

2. predict.m: 구해진 Theta값을 이용한 예측/판별 함수

3. sigmoid.m: Sigmoid 함수 구현체

+ 현재 regularization parameter, lambda,는 1로 설정되어 있는데 0일 경우에는 decision boundary가 중첩되면서 regularization term이 제 역할을 못하고 overfitting 
+ 100일 경우에는 underfitting으로 인해 theta zero만 남게됨