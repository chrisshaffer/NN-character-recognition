clear;
direc = 'C:\Users\Alex\Dropbox\EE210A_Project\EnglishHnd\English\Hnd\Img\Sample';%folder root location
class = [1:10];%class index
imnum = [55,55,55,55,55,55,55,55,55,55];%sample numbers in each calss


matindex = 1;
Data_matrix = zeros(550,65);
for i = 1:10;
    imindex = [1:imnum(i)];
    folder = sprintf('%03d',class(i));
    for j = 1:imnum(i)
        file = sprintf('%03d',imindex(j));
        data_rgb = imread(strcat(direc,folder,'\','img',folder,'-',file,'.png'));%read the picture based on class and imag number
        data_dig = imdigitalize(data_rgb);%transfer the rgb unsized image to 32*32 digitalized image with some denoising 
        data_oct = imdig2oct(data_dig);%transfer the 32*32 digital map into 0-16 scale 8*8 image
        data_vec = reshape(data_oct,[1,64]);
        Data_matrix(matindex,:) = [data_vec,i];%format the vector 
        matindex = matindex + 1;
    end
end

h = Data_matrix(:,1:64)'; % create matrix of input vectors for all samples

N = size(Data_matrix,1); % number of training samples
gamma = zeros(10,N);
for i = 1:N
    gamma(Data_matrix(i,65),i) = 1; % create matrix of class vectors
end

csvwrite('h_hwdigit.csv',h)
csvwrite('gamma_hwdigit.csv',gamma)