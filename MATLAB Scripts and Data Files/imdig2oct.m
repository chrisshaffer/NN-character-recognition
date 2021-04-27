function [data_q] = imdig2oct(data)
%input: 32*32 digitalized image map
%output:0-16 sacle 8*8 image
data_q = zeros(8,8);
for i = 1:8;
    for j = 1:8;
        data_q(i,j) = sum(sum(data(((i-1)*4+1):i*4,((j-1)*4+1):j*4)));
    end
end


end

