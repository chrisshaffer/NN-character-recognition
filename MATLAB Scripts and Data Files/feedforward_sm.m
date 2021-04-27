function [Z,Y] = feedforward_sm(W,theta,h,L,Nl)

% initialization of Z and Y
Z = zeros(max(Nl),L); % Z = before activation output
Y = zeros(max(Nl),L); % Y = after activation output

Y(1:Nl(1)) = h; % y1 = feature vector h

% Feedforward algorithm; see reference 47.968 on pg 2475
for l = 1:L-2
Z(1:Nl(l+1),l+1) = W(1:Nl(l+1),1:Nl(l),l)*Y(1:Nl(l),l)-theta(1:Nl(l+1),l);
Y(1:Nl(l+1),l+1) = softplus(Z(1:Nl(l+1),l+1)); % softplus activation function
end

Z(1:Nl(L),L) = W(1:Nl(L),1:Nl(L-1),L-1)*Y(1:Nl(L-1),L-1)-theta(1:Nl(L),L-1);
Y(1:Nl(L),L) = exp(Z(1:Nl(L),L))./(sum(exp(Z(1:Nl(L),L))));

% Note Y(1:Nl(L),L) = output of network
end