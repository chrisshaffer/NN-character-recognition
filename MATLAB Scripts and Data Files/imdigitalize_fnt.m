function [data_d] = imdigitalize_fnt(data)
%input is the rgb image with each pixel realize 0-255
%output is digitalized image with each pixel realize 0 or 1
data_gray = data;
data_gray3 = imresize(data_gray,[40 40]);
data_gray2 = data_gray3(5:36,5:36);%margin removal 
[M,N] = size(data_gray2);
thr = 40;%set the threshold for background removing
bgd = median([data_gray2(1,1),data_gray2(M,1),data_gray2(1,N),data_gray2(M,N)]);%get the bacground with a median filter
data_d = 1 - [(data_gray2 <= bgd + thr).*(data_gray2 >= bgd - thr)];%remove the backgroud and digitalize the image 

end

