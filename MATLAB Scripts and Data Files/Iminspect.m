function [] = Iminspect(H,Gamma,n)

List = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'];
h = H(:,n);
gamma = Gamma(:,n);
h_image = reshape(h,8,8);
imshow(h_image,[0,16])
indexmax = find(gamma == max(gamma));
display(List(indexmax))

end

