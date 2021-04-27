clear all
clc

L = 3; % number of layers

% regu = 2.7826e-4; % regularization factor
% stepsize = 0.006;
regu = 1e-4;
stepsize = 1e-3;
dir = 'C:\Users\macan\Dropbox\EE210A_Project\Summary_of_progress\';
% gam_file = 'gamma_hwdigit.csv';
% h_file = 'h_hwdigit.csv';
% gam_file = 'gamma_uppercasechar.csv';
% h_file = 'h_uppercasechar.csv';
% gam_file = 'gamma_hwucchar.csv';
% h_file = 'h_hwucchar.csv';
gam_file = 'gamma_ucchar_fnt.csv';
h_file = 'h_ucchar_fnt.csv';

Gamma = csvread([dir gam_file]); % matrix of all gamma vectors
H = csvread([dir h_file]); % matrix of all input vectors

imn = ones(1,26)*1016;
samplesize = 520*5;
testsize = round(samplesize*imn/sum(imn));

nend = sum(imn(1:1));
gammatrain = Gamma(:,(nend - imn(1)+1):(nend-testsize(1)));
gammatest = Gamma(:,(nend - testsize(1) +1):nend);
Gamma_train = gammatrain;
Gamma_test = gammatest;
htrain = H(:,(nend - imn(1)+1):(nend-testsize(1)));
htest = H(:,(nend - testsize(1) +1):nend);
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

n2 = 20:20:200;
final_performance = zeros(1, length(n2));    

tic
for l = 1:length(n2)
    Nl = [64,n2(l),26]; % number of nodes per layer
[W_old,theta_old] = network_initial_sp(L,Nl); % intitialization of W, theta with random values

for j = 1:10
rand_ind = randperm(size(Gamma_train,2));
for i = 1:size(Gamma_train,2)
%     Wmean(i) = mean(abs(W_old(W_old~=0)));
%     Wmax(i) = max(abs(W_old(W_old~=0)));
%     ind = randi(size(Gamma,2));
%     gamma = Gamma(:,ind);
%     h = H(:,ind);
    gamma = Gamma_train(:,rand_ind(i));
    h = H_train(:,rand_ind(i));
    [Z,Y] = feedforward_sm(W_old,theta_old,h,L,Nl);
    [W_new,theta_new] = grad_back_sm(Z,Y,L,Nl,regu,stepsize,gamma,W_old,theta_old);
    W_old = W_new; % update W for next iteration
    theta_old = theta_new; % update theta for next iteration
end
% subplot(2,1,1)
% plot(Wmean)
% subplot(2,1,2)
% plot(Wmax)
% end

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
end
final_performance(l) = performance(length(performance))*100;
end
toc
figure(1)
plot(n2,final_performance/100,'*');
% semilogx([final_performance(1:10)'; smooth(final_performance(11:length(final_performance)),50)],'*')
xlabel('n_2')
ylabel('Performance, 1-R_{test}')
grid on

Fsize = 16;
Msize = 10;
LW = 2;
fig=gcf;
set(findall(fig,'-property','FontSize'),'FontSize',Fsize)
set(findall(fig,'-property','MarkerSize'),'MarkerSize',Msize)
set(findall(fig,'-property','LineWidth'),'LineWidth',LW)
% plot(performance*100)
% xlabel('Iteration #')
% ylabel('Performance (%)')