function [y] = Dsigmoid(z)
%Compute the derivative of the sigmoid function
y = sigmoid(z).*(1-sigmoid(z));


end

