function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

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
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1: feedforwarding and cost function
I = eye(num_labels); % (10x10)
Y = zeros(m, num_labels); % (5000x10)
for i=1:m
  Y(i, :) = I(y(i), :);
end

a1 = [ones(m, 1) X];
z2 = a1*Theta1'; % (5000x401)*(401x25)=(5000x25)
a2 = [ones(size(z2, 1), 1) sigmoid(z2)]; % (5000x26)
z3 = a2*Theta2'; % (5000x26)*(26x10)=(5000x10)
h = sigmoid(z3); % (5000x10)

% Cost without regularization
J = sum(sum((-Y).*log(h) - (1-Y).*log(1-h)), 2)/m;
% Regularization penalty
Theta1_p = Theta1(:, 2:end);
Theta2_p = Theta2(:, 2:end);
p_1 = sum(sum(Theta1_p.^2, 2));
p_2 = sum(sum(Theta2_p.^2, 2));
p = lambda*(p_1 + p_2)/(2*m);
% Cost with regularization
J = J + p;

% Part 2: backpropagation and gradients
% Accumulate gradients using every example in the dataset
delta_1 = zeros(size(Theta1));
delta_2 = zeros(size(Theta2));

% This could be vectorized but was done through a for loop to show the operation
% over each training example
for t=1:m
  a1 = [1 X(t, :)]; % (1x401)
  z2 = a1*Theta1'; % (1x401)*(401x25)=(1x25)
  a2 = [1 sigmoid(z2)]; % (1x26)
  z3 = a2*Theta2'; % (1x26)*(26x10)=(1x10)
  a3 = sigmoid(z3);
  
  % Calculate sigmas
  % Layer 3
  sigma_k_3 = a3 - Y(t, :); % (1x10)
  % Layer 2
  sigma_k_2 = (sigma_k_3*Theta2) .* sigmoidGradient([1 z2]); % (1x10)*(10x26) .* (1x26)
  sigma_k_2 = sigma_k_2(2:end); % (1x25)
  
  % Accumulate gradients
  delta_1 = delta_1 + (sigma_k_2'*a1);
  delta_2 = delta_2 + (sigma_k_3'*a2);
end

% Compute regularization terms
p_1 = (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
p_2 = (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
Theta1_grad = delta_1./m + p_1;
Theta2_grad = delta_2./m + p_2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
