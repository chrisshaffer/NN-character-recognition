function [W_new,theta_new] = grad_back_ce(Z,Y,L,Nl,regu,stepsize,gamma,W_old,theta_old)
%See reference manual for details 
W_new = zeros(max(Nl),max(Nl),L-1); % W:combinational weights
theta_new = zeros(max(Nl),L-1); % theta:bias vector
delta = zeros(max(Nl),L); % the sensitive factor used in the computation

delta(1:Nl(L),L) = Y(1:Nl(L),L) - gamma;

for l = L-1:-1:1
    delta(1:Nl(l),l) = W_old(1:Nl(l+1),1:Nl(l),l)' * delta(1:Nl(l+1),l+1) .* Dsigmoid(Z(1:Nl(l),l));
    W_new(1:Nl(l+1),1:Nl(l),l) = (1-2*regu*stepsize)*W_old(1:Nl(l+1),1:Nl(l),l) - stepsize*delta(1:Nl(l+1),l+1)*Y(1:Nl(l),l)';
    theta_new(1:Nl(l+1),l) = theta_old(1:Nl(l+1),l) + stepsize*delta(1:Nl(l+1),l+1);
end

end