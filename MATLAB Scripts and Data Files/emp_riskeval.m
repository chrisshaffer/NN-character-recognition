function [Y,Y_result,emp_risk] = emp_riskeval(W,theta,H,L,Nl,Gamma)

[M,N] = size(Gamma);
Y = zeros(M,N);
Y_result = zeros(M,N);
for i = 1:N
    [Z,Y_temp] = feedforward(W,theta,H(:,i),L,Nl);
    Y(:,i) = Y_temp(1:M,L);
    Y_result(:,i) = maxchoose(Y(:,i));
end
emp_risk = 1;


end

