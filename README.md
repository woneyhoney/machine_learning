# machine_learning

< Assignment 1 - Linear Regression >

1. computeCost.m: 현재 주어진 X값과 정답 y값을 가지고 현재까지 구해진 theta값을 이용해서 최종 cost function 값인 J를 계산하는 함수

line 17: predictions = X * theta;
* 처음에 theta = [0; 0]으로 두고 주어진 m개의 training data에 대해 한 번에 predictions of hypothesis를 구함
* 그 다음 cost function J를 구함
* J는 output.txt에 쓰여짐
* 그 다음 theta = [-1; 2]로 두고 같은 방식으로 한 번 더 computeCost.m으로 J를 구하고 output.txt에 씀

2. gradientDescent.m: 현재 주어진 X값과 정답 y값 그리고 현재까지 구해진 theta값을 이용하고 이때 learning rate인 alpha값을 입력하으로 하여

 이후 theta값을 gradient descent 방식으로 계산하여 입력인 num_iters (iteration수)를 이용해서 J값의 변화 history를 vector형식으로 저장하는 함수

* theta = [0; 0]으로 두고 첫 번째 iteration에서 m개의 training data의 y값에 음수를 취한 값들의 평균 값이기에 learning rate와 곱해진 후 기존의 theta 값에서 빼주면서 다음 theta 값들이 update 됨 
* 정해진 iteration 수만큼 계속 돌면 J가 convex 중에서도 bowl-shaped이기 때문에 J가 local minimum에 빠지는 것 없이 
global minimam으로 수렴하도록 하는 theta가 정해질 것임
* 매 iteration마다 theta가 변하기 때문에 J값도 당연히 달라질 것이기에 이를 확인하고자 J_history에 J값을 저장함
* 마지막 iteration 후에 J를 최소화하는 theta 값을 output.txt에 씀  

3. output.txt (실행후 생성된 결과 파일)


< Assignment 2 - Regularized Logistic Regression >

1. costFunctionReg.m: Logistic Regression with Regularization에 대한 Cost Function

line 23: regCost = (lambda / (2 * m)) * norm(theta([2:end])) ^ 2;
* 1 - alpha * (lambda / m) = 0.99 로 가정 ( alpha = learning rate, lambda = regularization parameter, m = no. of traning examples) 
* Gradient Descent의 one iteration에서 theta가 0.99배가 되니 square norm oftheta가 조금 더 작아지는 것을 반영하기 위해 norm(theta([2:end])) ^ 2

2. predict.m: 구해진 Theta값을 이용한 예측/판별 함수

3. sigmoid.m: Sigmoid 함수 구현체

+ 현재 regularization parameter, lambda,는 1로 설정되어 있는데 0일 경우에는 decision boundary가 중첩되면서 regularization term이 제 역할을 못하고 overfitting 
+ 100일 경우에는 underfitting으로 인해 theta zero만 남게됨


< Assignment 3 - 3 Layer Neural Network > 

- Classification Task: 1개의 학습 Sample에는 20x20 pixel 기반의 Gray 숫자 영상 -> 400x1 vector로 표현됨
- 상기 네트워크는 중간에 Hidden Layer로 총 25개의 Unit이 존재하고 Bias가 존재 
- 총 m = 5000개의 학습 Sample이 제공되며 최종 학습 Data는 5000 x 200 크기의 X matrix
- 최종 결과: 0~9로 표현 되나 여기서는 편의를 위해 1,2,3,...,9,0 으로 표현. 
- 따라서, 본 과제에서는 Class의 개수가 10이 됨
- nnCostFunction.m에는 ForwardPropagation과 BackPropagation 함수를 호출해서 J값과 Gradient 값을 출력해줌
- Gradient Descent 학습: 총 50번의 Iteration을 통해서 학습을 수행 (당연히 매 Iteration마다 Cost값은 감소 해야함)
- 최종 목표: 성능 약 95% 이상 (prediction.m으로 학습 Data에 대한 성능 평가)

1. BackPropagation.m: Backpropagation 함수 제작 

+ 본 과제에서 Weight의 Random Initialization 값은 기존 코드를 그대로 사용함 
+ 본 과제에서 gradient checking은 사용하지 않음

2. FowardPropagation.m: Fowardpropagation 함수 제작

3. sigmoidGradient.m: Sigmoid 미분 함수 제작