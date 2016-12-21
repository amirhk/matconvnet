% --------------------------------------------------------------------
function imdb = constructMnistImdb(opts)
% --------------------------------------------------------------------
  afprintf(sprintf('[INFO] Constructing MNIST imdb...\n'));
  % Preapre the imdb structure, returns image data with mean image subtracted
  files = {'train-images-idx3-ubyte', ...
           'train-labels-idx1-ubyte', ...
           't10k-images-idx3-ubyte', ...
           't10k-labels-idx1-ubyte'};

  if ~exist(opts.imdb.data_dir, 'dir')
    mkdir(opts.imdb.data_dir);
  end

  for i=1:4
    if ~exist(fullfile(opts.imdb.data_dir, files{i}), 'file')
      url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i});
      afprintf(sprintf('downloading %s\n', url));
      gunzip(url, opts.imdb.data_dir);
    end
  end

  f=fopen(fullfile(opts.imdb.data_dir, 'train-images-idx3-ubyte'),'r');
  x1=fread(f,inf,'uint8');
  fclose(f);
  x1=permute(reshape(x1(17:end),28,28,60e3),[2 1 3]);

  f=fopen(fullfile(opts.imdb.data_dir, 't10k-images-idx3-ubyte'),'r');
  x2=fread(f,inf,'uint8');
  fclose(f);
  x2=permute(reshape(x2(17:end),28,28,10e3),[2 1 3]);

  f=fopen(fullfile(opts.imdb.data_dir, 'train-labels-idx1-ubyte'),'r');
  y1=fread(f,inf,'uint8');
  fclose(f);
  y1=double(y1(9:end)')+1;

  f=fopen(fullfile(opts.imdb.data_dir, 't10k-labels-idx1-ubyte'),'r');
  y2=fread(f,inf,'uint8');
  fclose(f);
  y2=double(y2(9:end)')+1;

  set = [ones(1,numel(y1)) 3*ones(1,numel(y2))];
  if strcmp(opts.general.network_arch, 'mnistnet')
    data = single(reshape(cat(3, x1, x2),28,28,1,[]));
  elseif strcmp(opts.general.network_arch, 'lenet')
    % MNIST is single channel of size 28x28.. for LeNet, repmat channel 3 and pad
    data = single(padarray(repmat(reshape(cat(3, x1, x2),28,28,1,[]), [1,1,3,1]), [2,2]));
  else
    disp('wtf!!')
  end
  data_mean = mean(data(:,:,:,set == 1), 4);
  data = bsxfun(@minus, data, data_mean);

  imdb.images.data = data;
  imdb.images.data_mean = data_mean;
  imdb.images.labels = cat(2, y1, y2);
  imdb.images.set = set;
  imdb.meta.sets = {'train', 'val', 'test'};
  imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false);
  afprintf(sprintf('done!\n\n'));
