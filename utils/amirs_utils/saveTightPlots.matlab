
make_it_tight = true;
subplot = @(m,n,p) subtightplot (m, n, p, [0.01 0.05], [0.1 0.01], [0.1 0.01]);
if ~make_it_tight,  clear subplot;  end


% figure,
% for i = 1:4
%   a = original_imdb.images.data(:,:,1,i+20);
%   a(a==1) = 0;
%   a(a==0.4) = 1;
%   subplot(2,2,i)
%   imshow(a)
% end

% figure,
% for i = 1:16
%   a = original_imdb.images.data(:,:,:,i+200);
%   a(a==1) = 0;
%   a(a==0.4) = 1;
%   subplot(4,4,i)
%   imshow()
% end

% figure,
% for i = 1:16
%   a = original_imdb.images.data(:,:,:,i+100);
%   subplot(4,4,i)
%   imshow(imrotate(a,270))
% end

figure,
for i = 1:16
  a = original_imdb.images.data(:,:,:,i+100);
  subplot(4,4,i)
  imshow(a,[])
end






print(sprintf('cifar-all'),'-depsc')
print(sprintf('imagenet-tiny-all'),'-depsc')
