clear all
clc

L = 3; % number of layers
Nl = [64,30,26]; % number of nodes per layer
regu = 10e-3; % regularization factor
stepsize = 1e-3;

dir = 'C:\Users\Alex\Dropbox\EE210A_Project\Summary_of_progress\';
% gam_file = 'gamma_hwdigit.csv';
% h_file = 'h_hwdigit.csv';
% gam_file = 'gamma_uppercasechar.csv';
% h_file = 'h_uppercasechar.csv';
gam_file = 'gamma_hwucchar.csv';
h_file = 'h_hwucchar.csv';

Gamma = csvread([dir gam_file]); % matrix of all gamma vectors
H = csvread([dir h_file]); % matrix of all input vectors


imn = 55*ones(1,26);%sample numbers in each calss
samplesize = 260;
testsize = round(samplesize*imn/sum(imn));

nend = sum(imn(1:1));
gammatrain = Gamma(:,(nend - imn(1)+1):(nend-testsize(1)));
gammatest = Gamma(:,(nend - testsize(1) +1):nend);
Gamma_train = gammatrain;
Gamma_test = gammatest;
htrain = H(:,(nend - imn(1)+1):(nend-testsize(1)));
htest =H(:,(nend - testsize(1) +1):nend);
H_train = htrain;
H_test = htest;

for i = 2:26
    nend = sum(imn(1:i));
    gammatrain = Gamma(:,(nend - imn(i)+1):(nend-testsize(i)));
    gammatest = Gamma(:,(nend - testsize(i) +1):nend);
    Gamma_train = [Gamma_train,gammatrain];
    Gamma_test = [Gamma_test,gammatest];
    htrain = H(:,(nend - imn(i)+1):(nend-testsize(i)));
    htest =H(:,(nend - testsize(i) +1):nend);    
    H_train = [H_train,htrain];
    H_test = [H_test,htest;];
end


    

[W_old,theta_old] = network_initial_sp(L,Nl); % intitialization of W, theta with random values
tic
for j = 1:500
rand_ind = randperm(size(Gamma_train,2));
for i = 1:size(Gamma_train,2)

    gamma = Gamma_train(:,rand_ind(i));
    h = H_train(:,rand_ind(i));
    [Z,Y] = feedforward_sp(W_old,theta_old,h,L,Nl);
    [W_new,theta_new] = grad_back_sp(Z,Y,L,Nl,regu,stepsize,gamma,W_old,theta_old);
    W_old = W_new; % update W for next iteration
    theta_old = theta_new; % update theta for next iteration
end


% 
% Testing on training data
gamma_est = zeros(Nl(L),1);
correct = 0;
performance = zeros(size(Gamma_train,2),1);
for i = 1:size(Gamma_train,2)
    gamma = Gamma_train(:,i);
    h = H_train(:,i);
    [Z,Y] = feedforward_sp(W_old,theta_old,h,L,Nl); % calculate Z, Y matrices
    y = Y(1:Nl(L),L); % extract output from Y matrix
    gamma_est = maxchoose(y); % convert y into class vector
    correct = correct + isequal(gamma,gamma_est); % compare calculated class vector with actual class vector
    performance(i) = correct/i; % update performance 
end
final_performance(j) = performance(length(performance))*100;
end
toc
figure(1)
semilogx(smooth(final_performance,10),'*')
xlabel('Run #')
ylabel('Performance (%)')
title('3 layers')


