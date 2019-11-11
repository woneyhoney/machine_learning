function [J] = FowardPropagation(nn_params, ...
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
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed

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

% h(x), 5000 x 10 = (# of training examples) x (# of class)
h = a3;

% Calculate cost J 
J = (-1/m) * sum(sum(Y .* log(h) + ((1-Y) .* log(1-h))));

%
% Part 2: Implement regularization with the cost function
%

s1 = input_layer_size;  % 400
s2 = hidden_layer_size; % 25
s3 = k;                 % 10

% remove bias units 
t1 = Theta1(:, 2:end);  % 25 * 400
t2 = Theta2(:, 2:end);  % 10 * 25

regTerm = (lambda / (2 * m)) * (sum(sum(t1 .^ 2)) + sum(sum(t2 .^ 2)));

J = J + regTerm;

% =============================================================

end

