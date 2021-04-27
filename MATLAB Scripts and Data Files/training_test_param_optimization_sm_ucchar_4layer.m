clear all
clc

L = 4; % number of layers
Nl = [64,100,60,26]; % number of nodes per layer
regu = logspace(-8,0,10); % regularization factor
stepsize = logspace(-4,-2,10);

dir = 'C:\Users\Shuyang Jiang\Dropbox\EE210A_Project\Summary_of_progress\';
% gam_file = 'gamma_hwdigit.csv';
% h_file = 'h_hwdigit.csv';
gam_file = 'gamma_uppercasechar.csv';
h_file = 'h_uppercasechar.csv';
% gam_file = 'gamma_hwucchar.csv';
% h_file = 'h_hwucchar.csv';

Gamma = csvread([dir gam_file]); % matrix of all gamma vectors
H = csvread([dir h_file]); % matrix of all input vectors

imn = [558,115,215,191,446,79,143,193,302,77,92,215,149,363,382,159,35,389,342,312,92,84,67,80,67,55];
samplesize = 500;
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

final_performance = zeros(length(stepsize), length(regu));

tic
for l = 1:length(stepsize)
    for k = 1:length(regu)
        [W_old,theta_old] = network_initial_sp(L,Nl); % intitialization of W, theta with random values
        for j = 1:10
            rand_ind = randperm(size(Gamma_train,2));
            for i = 1:size(Gamma_train,2)
                gamma = Gamma_train(:,rand_ind(i));
                h = H_train(:,rand_ind(i));
                [Z,Y] = feedforward_sm(W_old,theta_old,h,L,Nl);
                [W_new,theta_new] = grad_back_sm(Z,Y,L,Nl,regu(k),stepsize(l),gamma,W_old,theta_old);
                W_old = W_new; % update W for next iteration
                theta_old = theta_new; % update theta for next iteration
            end

            
        %
        % Testing on training data
        gamma_est = zeros(Nl(L),1);
        correct = 0;
        performance = zeros(size(Gamma_test,2),1);
        for i = 1:size(Gamma_test,2)
            gamma = Gamma_test(:,i);
            h = H_test(:,i);
            [Z,Y] = feedforward_sm(W_old,theta_old,h,L,Nl); % calculate Z, Y matrices
            y = Y(1:Nl(L),L); % extract output from Y matrix
            gamma_est = maxchoose(y); % convert y into class vector
            correct = correct + isequal(gamma,gamma_est); % compare calculated class vector with actual class vector
            performance(i) = correct/i; % update performance
        end
        final_performance(l,k) = performance(length(performance))*100;
        end
        k
    end
    l
end
toc


surfc(stepsize,regu,(final_performance/100)');
set(gca, 'XScale', 'log', 'YScale', 'log')
xlabel('$\mu$','Interpreter','latex')
ylabel('$\rho$','Interpreter','latex')
c = sprintf('Max performance = %2.2f',max(max(final_performance/100)))
title(c);
g = colorbar;
ylabel(g, 'Performance, 1 - R_{test}')
az = 0;
el = 90;
view(az, el);
Fsize = 16;
Msize = 10;
LW = 2;
fig=gcf;
set(findall(fig,'-property','FontSize'),'FontSize',Fsize)
set(findall(fig,'-property','MarkerSize'),'MarkerSize',Msize)
set(findall(fig,'-property','LineWidth'),'LineWidth',LW)