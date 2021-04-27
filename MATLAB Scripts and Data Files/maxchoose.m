function [y] = maxchoose(x)

y = x.* 0;
indexmax = find(x == max(x));
y(indexmax) = 1;



end

