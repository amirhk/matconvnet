% % -------------------------------------------------------------------------
% function imdb = constructSTL10Imdb(opts)
% % -------------------------------------------------------------------------
%   fprintf('[INFO] Constructing STL-10 imdb...');
%   train_file = load(fullfile(opts.dataDir, 'train.mat'));
%   test_file = load(fullfile(opts.dataDir, 'test.mat'));

%   % data_train = imresize(reshape(train_file.X', 96,96,3,[]), [32,32]);
%   data_train = imresize(reshape(im2double(train_file.X'), 96,96,3,[]), [32,32]);
%   labels_train = single(train_file.y');
%   set_train = 1 * ones(1, 5000);

%   % data_test = imresize(reshape(test_file.X', 96,96,3,[]), [32,32]);
%   data_test = imresize(reshape(im2double(test_file.X'), 96,96,3,[]), [32,32]);
%   labels_test = single(test_file.y');
%   set_test = 3 * ones(1, 8000);

%   data = single(cat(4, data_train, data_test));
%   labels = single(cat(2, labels_train, labels_test));
%   set = cat(2, set_train, set_test);

%   % remove mean in any case
%   dataMean = mean(data(:,:,:,set == 1), 4);
%   data = bsxfun(@minus, data, dataMean);

%   if opts.contrastNormalization
%     fprintf('[INFO] contrast-normalizing data... ');
%     z = reshape(data,[],13000);
%     z = bsxfun(@minus, z, mean(z,1));
%     n = std(z,0,1);
%     z = bsxfun(@times, z, mean(n) ./ max(n, 40));
%     data = reshape(z, 32, 32, 3, []);
%     fprintf('done.\n');
%   end

%   % if opts.whitenData
%   %   fprintf('[INFO] whitening data... ');
%   %   z = reshape(data,[],13000);
%   %   W = z(:,set == 1)*z(:,set == 1)'/13000;
%   %   [V,D] = eig(W);
%   %   % the scale is selected to approximately preserve the norm of W
%   %   d2 = diag(D);
%   %   en = sqrt(mean(d2));
%   %   z = V*diag(en./max(sqrt(d2), 10))*V'*z;
%   %   data = reshape(z, 32, 32, 3, []);
%   %   fprintf('done.\n');
%   % end

%   imdb.images.data = data;
%   imdb.images.labels = labels;
%   imdb.images.set = set;
%   imdb.meta.sets = {'train', 'val', 'test'};
%   imdb.meta.classes = train_file.class_names; % = test_file.class_names
%   fprintf('done!\n\n');


% --------------------------------------------------------------------
function imdb = constructSTL10Imdb(opts)
ImgSize = 32;
Portion = 1;
% --------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
tmp = load(fullfile(opts.dataDir,'train.mat'));
data_train = im2double(tmp.X(1:end*Portion,:));%%%%%%%%%%%%%%%%
data_train = permute(reshape(data_train,[size(data_train,1) 96 96 3]), [2 3 4 1]);%%%%%%%%%%%%%%%%%%
if (ImgSize ~=96)
    data_train1 = [];
    for i =1 :size(data_train,4)
        data_train1(:,:,:,i) = imresize(data_train(:,:,:,i), [ImgSize ImgSize]);
    end
    data_train = data_train1;
end

labels_train = tmp.y(1:end*Portion); % Index from 1
set_train = ones(length(labels_train),1);


tmp = load(fullfile(opts.dataDir,'test.mat'));
data_test = im2double(tmp.X);
data_test = permute(reshape(data_test,[8000 96 96 3]), [2 3 4 1]);

if (ImgSize ~=96)
    data_test1 = [] ;
    for i =1 :size(data_test,4)
        data_test1(:,:,:,i) = imresize(data_test(:,:,:,i), [ImgSize ImgSize]);
    end
    data_test = data_test1;
end


labels_test = tmp.y; % Index from 1
set_test = 3*ones(length(labels_test),1);


% remove mean in any case
dataMean = mean(data_train, 4);

data = (cat(4,data_train,data_test));
%--------------------------------------------------------------------------
% mean subtraction
% data = bsxfun(@minus, data, dataMean);
%--------------------------------------------------------------------------
% datatmp = zeros(size(data,1),size(data,2),size(data,3),size(data,4));
% for i = 1:size(data,4)
%     datatmp(:,:,1:3,i) =rgb2ycbcr( data(:,:,:,i));
% end
% data = datatmp(:,:,3,:);
%--------------------------------------------------------------------------
        % Local contrast normalization
% h1 = (1/81).*ones(9,9);
% h1 = fspecial('gaussian', [5 5] , 1);
% h2 = ones(9,9);
% for i = 1:size(data,4)
%     LM(:,:,:,i) = imfilter(data(:,:,:,i),h1);
% %     LS(:,:,:,i) = stdfilt(data(:,:,:,i),h2);
% end
% data = ( LM);
% % % data = (data - LM)./(LS.^2);
% clear LM
% clear LS
% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng

data = single(data);
z  = reshape(data, [ImgSize*ImgSize*3,size(data,4)]); %%%%%%%%%%%%%%%%%%%%%%%%%%%%
if opts.contrastNormalization
%   z = reshape(data,[],size(data,4)) ;
  z = bsxfun(@minus, z, mean(z,2)) ;
%   n = std(z,0,2) ;
%   z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
  data = reshape(z, ImgSize, ImgSize, 3, []) ; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
set = [set_train; set_test]';
if opts.whitenData
  z = reshape(data,[],size(data,4)) ;
  W = z(:,set == 1)*z(:,set == 1)'/size(data,4) ;
  clear data;
  [V,D] = eig(W) ;
  % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ;
  en = sqrt(mean(d2)) ;
  z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
  data = reshape(z, ImgSize, ImgSize, 3, []) ;
end

imdb.images.data = data;
imdb.images.labels   = cat(1, labels_train, labels_test)';
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
