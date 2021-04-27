function [W,theta] = network_initial(L,Nl)

W = zeros(max(Nl),max(Nl),L-1); % Initialize weights as zeros
theta = zeros(max(Nl),L-1); % Initialize bias vector as zeros

for l = 1:L-1
%     W(1:Nl(l+1),1:Nl(l),l) = normrnd(0,1/sqrt(Nl(l)),Nl(l+1),Nl(l)); % Gaussian distribution with mean=0, variance=1/nl
    W(1:Nl(l+1),1:Nl(l),l) = (rand(Nl(l+1),Nl(l))-.5)*2*4*sqrt(6)/sqrt(Nl(l+1)+Nl(l)); % Uniform distribution over range +-4*sqrt(6)/sqrt(Nl(l+1)+Nl(l))
%     W(1:Nl(l+1),1:Nl(l),l) = (rand(Nl(l+1),Nl(l))-.5)*2/sqrt(Nl(l)); % Uniform distribution over range +-1/sqrt(Nl(l))

theta(1:Nl(l+1),l) = normrnd(0,1,Nl(l+1),1); % Gaussian distribution with mean=0, variance=1
end

end