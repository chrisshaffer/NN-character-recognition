gamma_est = zeros(Nl(L),1);
correct = 0;
performance = zeros((size(Gamma_test,2)),1);
correctchar = zeros((size(Gamma_test,2)),1);
for i = 1:(size(Gamma_test,2))
    gamma = Gamma_test(:,i);
    h = H_test(:,i);
    [Z,Y] = feedforward_sm(W_old,theta_old,h,L,Nl); % calculate Z, Y matrices
    y = Y(1:Nl(L),L); % extract output from Y matrix
    gamma_est = maxchoose(y); % convert y into class vector
    correct = correct + isequal(gamma,gamma_est); % compare calculated class vector with actual class vector
    performance(i) = correct/i; % update performance 
    correctchar(i) = correct;
end
final_performance = performance(length(performance))*100


ncorrect = zeros(1,26);
for n = 1:26
    if n == 1
         ncorrect(n) = correctchar(testsize(1));
    else
         ncorrect(n) = correctchar(sum(testsize(1:n))) - correctchar(sum(testsize(1:n-1)));
    end
end
n = 1:26;
N = char(n+64);

figure;
stem(ncorrect./testsize)
c = sprintf('Rate of Correct Recognition = %2.2f%%',final_performance)
xlabel(c)
ylabel('Performance, 1-R_{test}')
set(gca,'XTick',n,'XTickLabel',{'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'})