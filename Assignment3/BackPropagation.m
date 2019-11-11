function [grad] = BackPropagation(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly 
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 3: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

k = num_labels; % 10

% initialize Y as the classfication output
diagonal = eye(k);
Y = diagonal(y, :);

% make each row of X have a bias unit in layer 1
bias = ones(m,1);
a1 = [bias X]; 

% input of layer 2
z2 = a1 * Theta1';
% output of layer 2 with bias units added 
a2 = [bias sigmoid(z2)];

% input of layer 3
z3 = a2 * Theta2';
% output of layer 3 
a3 = sigmoid(z3);

% layer 3 errors, 5000 x 10
delta3 = a3 - Y;
% layer 2 errors, 5000 x 26
delta2 = (delta3 * Theta2) .* [ones(size(z2,1),1) sigmoidGradient(z2)];
% remove delta2 for bias node, 5000 x 25
delta2 = delta2(:,2:end);

% 25 x 401, partial derivative of layer 1 without the reg term 
Theta1_grad = (1/m) * (delta2' * a1);
% 10 x 26, partial derivative of layer 2 without the reg term 
Theta2_grad = (1/m) * (delta3' * a2);

% Part 4: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 3.

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda / m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda / m) * Theta2(:, 2:end);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

