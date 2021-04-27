function [f] = sigmoid(z)

f = 1./(1+exp(-z));

end