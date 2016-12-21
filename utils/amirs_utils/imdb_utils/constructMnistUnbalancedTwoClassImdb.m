% --------------------------------------------------------------------
function imdb = constructMnistUnbalancedTwoClassImdb(opts)
% --------------------------------------------------------------------
  afprintf(sprintf('[INFO] Constructing MNIST imdb...\n'));
  % Preapre the imdb structure, returns image data with mean image subtracted
  files = {'train-images-idx3-ubyte', ...
           'train-labels-idx1-ubyte', ...
           't10k-images-idx3-ubyte', ...
           't10k-labels-idx1-ubyte'} ;

  if ~exist(opts.imdb.dataDir, 'dir')
    mkdir(opts.imdb.dataDir) ;
  end

  for i=1:4
    if ~exist(fullfile(opts.imdb.dataDir, files{i}), 'file')
      url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
      fprintf('downloading %s\n', url) ;
      gunzip(url, opts.imdb.dataDir) ;
    end
  end

  f=fopen(fullfile(opts.imdb.dataDir, 'train-images-idx3-ubyte'),'r') ;
  x1=fread(f,inf,'uint8');
  fclose(f) ;
  x1=permute(reshape(x1(17:end),28,28,60e3),[2 1 3]) ;

  f=fopen(fullfile(opts.imdb.dataDir, 't10k-images-idx3-ubyte'),'r') ;
  x2=fread(f,inf,'uint8');
  fclose(f) ;
  x2=permute(reshape(x2(17:end),28,28,10e3),[2 1 3]) ;

  f=fopen(fullfile(opts.imdb.dataDir, 'train-labels-idx1-ubyte'),'r') ;
  y1=fread(f,inf,'uint8');
  fclose(f) ;
  y1=double(y1(9:end)')+1 ;

  f=fopen(fullfile(opts.imdb.dataDir, 't10k-labels-idx1-ubyte'),'r') ;
  y2=fread(f,inf,'uint8');
  fclose(f) ;
  y2=double(y2(9:end)')+1 ;

  set = [ones(1,numel(y1)) 3*ones(1,numel(y2))];
  if strcmp(opts.general.networkArch, 'mnistnet')
    data = single(reshape(cat(3, x1, x2),28,28,1,[]));
  elseif strcmp(opts.general.networkArch, 'lenet')
    % MNIST is single channel of size 28x28.. for LeNet, repmat channel 3 and pad
    data = single(padarray(repmat(reshape(cat(3, x1, x2),28,28,1,[]), [1,1,3,1]), [2,2]));
  else
    disp('wtf!!')
  end
  dataMean = mean(data(:,:,:,set == 1), 4);
  data = bsxfun(@minus, data, dataMean) ;

  imdb.images.data = data ;
  imdb.images.data_mean = dataMean;
  imdb.images.labels = cat(2, y1, y2) ;
  imdb.images.set = set ;
  imdb.meta.sets = {'train', 'val', 'test'} ;
  imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;
  % fprintf('done!\n\n');

  data_ones = imdb.images.data(:,:,:, imdb.images.labels == 1);
  subsampled_data_ones_indices = randsample(size(data_ones, 4), floor(size(data_ones, 4) / 500));
  subsampled_data_ones = data_ones(:,:,:, subsampled_data_ones_indices);
  data_nines = imdb.images.data(:,:,:, imdb.images.labels == 9);

  data_positive = subsampled_data_ones;
  labels_positive = 2 * ones(1, size(data_positive, 4));
  data_negative = data_nines;
  labels_negative = 1 * ones(1, size(data_negative, 4));

  two_class_data = cat(4, data_positive, data_negative);
  two_class_labels = cat(2, labels_positive, labels_negative);

  % imdb.images.data = data ;
  % imdb.images.data_mean = dataMean;
  % imdb.images.labels = cat(2, y1, y2) ;
  imdb.images.data = two_class_data;
  imdb.images.labels = two_class_labels;
  % imdb.images.set = set ;
  imdb.images.set = (round(rand(1,length(two_class_labels))) * 2) + 1; % randomly assign to either set 1 or set 3
  imdb.meta.sets = {'train', 'val', 'test'} ;
  imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;
  afprintf(sprintf('done!\n\n'));





  afprintf(sprintf('== == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==\n\n'));

  % afprintf(sprintf('[INFO] total number of samples: %d\n', totalNumberOfSamples));
  % afprintf(sprintf('[INFO] number of `train` data - negative: %d\n', size(data_train(:,:,:,labels_train == 1),4)));
  % afprintf(sprintf('[INFO] number of `train` data - positive: %d\n', size(data_train(:,:,:,labels_train == 2),4)));
  % afprintf(sprintf('[INFO] number of `test` data - negative: %d\n', size(data_test(:,:,:,labels_test == 1),4)));
  % afprintf(sprintf('[INFO] number of `test` data - positive: %d\n', size(data_test(:,:,:,labels_test == 2),4)));
  afprintf( ...
    sprintf( ...
      '[INFO] number of `train` data - negative: %d\n', ...
      size( ...
        imdb.images.data(:,:,:, ...
          bsxfun(@and, imdb.images.labels == 1, imdb.images.set == 1)), ...
    4)));
  afprintf( ...
    sprintf( ...
      '[INFO] number of `train` data - positive: %d\n', ...
      size( ...
        imdb.images.data(:,:,:, ...
          bsxfun(@and, imdb.images.labels == 2, imdb.images.set == 1)), ...
    4)));
  afprintf( ...
    sprintf( ...
      '[INFO] number of `test` data - negative: %d\n', ...
      size( ...
        imdb.images.data(:,:,:, ...
          bsxfun(@and, imdb.images.labels == 1, imdb.images.set == 3)), ...
    4)));
  afprintf( ...
    sprintf( ...
      '[INFO] number of `test` data - positive: %d\n', ...
      size( ...
        imdb.images.data(:,:,:, ...
          bsxfun(@and, imdb.images.labels == 2, imdb.images.set == 3)), ...
    4)));










































