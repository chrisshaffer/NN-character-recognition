direc = 'C:\Users\Alex\Dropbox\EE210A_Project\Chars74K_dataset\English\Img\GoodImg\Bmp\Sample';%folder root location
class = [11:36];%class index
imnum = [558,115,215,191,446,79,143,193,302,77,92,215,149,363,382,159,35,389,342,312,92,84,67,80,67,55];%sample numbers in each calss
%         A   B   C   D   E  F   G   H   I  J  K   L   M   N   O   P   Q
%    R   S   T  U  V  W  X  Y  Z

matindex = 1;
Data_matrix = zeros(5202,65);
for i = 1:26;
    imindex = [1:imnum(i)];
    folder = sprintf('%03d',class(i));
    for j = 1:imnum(i)
        file = sprintf('%03d',imindex(j));
        data_rgb = imread(strcat(direc,folder,'\','img',folder,'-00',file,'.png'));%read the picture based on class and imag number
        data_dig = imdigitalize(data_rgb);%transfer the rgb unsized image to 32*32 digitalized image with some denoising 
        data_oct = imdig2oct(data_dig);%transfer the 32*32 digital map into 0-16 scale 8*8 image
        data_vec = reshape(data_oct,[1,64]);
        Data_matrix(matindex,:) = [data_vec,i];%format the vector 
        matindex = matindex + 1;
    end
end

h = Data_matrix(:,1:64)'; % create matrix of input vectors for all samples

N = size(Data_matrix,1); % number of training samples
gamma = zeros(26,N);
for i = 1:N
    gamma(Data_matrix(i,65),i) = 1; % create matrix of class vectors
end

csvwrite('h_uppercasechar.csv',h)
csvwrite('gamma_uppercasechar.csv',gamma)