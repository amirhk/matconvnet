function [net, info] = cnn_amir(varargin)
  run(fullfile(fileparts(mfilename('fullpath')), ...
    '..', 'matlab', 'vl_setupnn.m'));

  % Setup -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
  opts.train = struct();
  opts.networkType = 'alex-net';
  opts.dataset = 'cifar';
  opts.backpropDepth = 20;
  opts.weightDecay = 0.0001;
  opts.weightInitType = '1D';
  opts.weightInitSource = 'load';
  [opts, varargin] = vl_argparse(opts, varargin);
  fprintf('[INFO] networkType:\t %s\n', opts.networkType);
  fprintf('[INFO] dataset:\t\t %s\n', opts.dataset);
  fprintf('[INFO] backpropDepth:\t %d\n', opts.backpropDepth);
  fprintf('[INFO] weightDecay:\t %s\n', opts.weightDecay);
  fprintf('[INFO] weightInitType:\t %s\n', opts.weightInitType);
  fprintf('[INFO] weightInitSource: %s\n', opts.weightInitSource);
  fprintf('\n');

  % Processor -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  [opts.train.gpus, opts.processorString] = getProcessor(opts);
  [opts, varargin] = vl_argparse(opts, varargin);

  % Paths -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
  opts.timeString = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.imdbDir = fullfile(vl_rootnn, 'data', sprintf( ...
    '%s-%s', ...
    opts.dataset, ...
    opts.networkType));
  opts.expDir = fullfile(vl_rootnn, 'data', sprintf( ...
    '%s-%s-%s-%s', ...
    opts.dataset, ...
    opts.networkType, ...
    opts.timeString, ...
    opts.processorString));
  [opts, varargin] = vl_argparse(opts, varargin);
  if ~exist(opts.expDir)
    mkdir(opts.expDir);
  end
  opts.dataDir = fullfile(vl_rootnn, 'data', opts.dataset);
  opts.imdbPath = fullfile(opts.imdbDir, 'imdb.mat');

  % Other -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
  opts.whitenData = true;
  opts.contrastNormalization = true;
  opts = vl_argparse(opts, varargin);

  % -------------------------------------------------------------------------
  %                                                    Prepare model and data
  % -------------------------------------------------------------------------
  net = cnn_amir_init( ...
    'weightDecay', opts.weightDecay, ...
    'weightInitType', opts.weightInitType, ...
    'weightInitSource', opts.weightInitSource, ...
    'backpropDepth', opts.backpropDepth, ...
    'networkType', opts.networkType, ...
    'dataset', opts.dataset);
  saveNetworkInfo(net, opts.expDir);

  if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath);
  else
    switch opts.dataset
      case 'cifar'
        imdb = getCifarImdb(opts);
      case 'mnist'
        imdb = getMnistImdb(opts);
    end
    mkdir(opts.expDir);
    save(opts.imdbPath, '-struct', 'imdb');
  end

  net.meta.classes.name = imdb.meta.classes(:)';

  % -------------------------------------------------------------------------
  %                                                                     Train
  % -------------------------------------------------------------------------
  [net, info] = cnn_train(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, ...
    net.meta.trainOpts, ...
    opts.train, ...
    'val', find(imdb.images.set == 3));

% -------------------------------------------------------------------------
function [processorList, processorString] = getProcessor(opts)
% -------------------------------------------------------------------------
  if ~isfield(opts.train, 'gpus')
    if ispc
      % freeGPUIndex = getFreeGPUIndex();
      freeGPUIndex = 1;
      if freeGPUIndex ~= -1
        processorList = [freeGPUIndex];
        processorString = sprintf('GPU%d', freeGPUIndex);
      else
        processorString = 'CPU';
        processorList = [];
      end
    else
      processorString = 'CPU';
      processorList = [];
    end
  end;

% -------------------------------------------------------------------------
function randomGPUIndex = saveNetworkInfo(net, expDir)
% -------------------------------------------------------------------------
  fprintf('\n[INFO] Saving network info in readme...\n');
  struct2File( ...
    net.meta.trainOpts, ...
    fullfile(expDir, 'readme.txt'), ...
    'delimiter', ...
    '\n\n');

% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
  fn = @(x,y) getSimpleNNBatch(x,y);

% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
  images = imdb.images.data(:,:,:,batch);
  labels = imdb.images.labels(1,batch);
  if rand > 0.5, images=fliplr(images); end

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
  images = imdb.images.data(:,:,:,batch);
  labels = imdb.images.labels(1,batch);
  if rand > 0.5, images=fliplr(images); end
  if opts.numGpus > 0
    images = gpuArray(images);
  end
  inputs = {'input', images, 'label', labels};

% -------------------------------------------------------------------------
function imdb = getCifarImdb(opts)
% -------------------------------------------------------------------------
  % Prepare the imdb structure, returns image data with mean image subtracted
  unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
  files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
    {'test_batch.mat'}];
  files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
  file_set = uint8([ones(1, 5), 3]);

  if any(cellfun(@(fn) ~exist(fn, 'file'), files))
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
    fprintf('downloading %s\n', url);
    untar(url, opts.dataDir);
  end

  data = cell(1, numel(files));
  labels = cell(1, numel(files));
  sets = cell(1, numel(files));
  for fi = 1:numel(files)
    fd = load(files{fi});
    data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]);
    labels{fi} = fd.labels' + 1; % Index from 1
    sets{fi} = repmat(file_set(fi), size(labels{fi}));
  end

  set = cat(2, sets{:});
  data = single(cat(4, data{:}));

  % remove mean in any case
  dataMean = mean(data(:,:,:,set == 1), 4);
  data = bsxfun(@minus, data, dataMean);

  % normalize by image mean and std as suggested in `An Analysis of
  % Single-Layer Networks in Unsupervised Feature Learning` Adam
  % Coates, Honglak Lee, Andrew Y. Ng

  if opts.contrastNormalization
    z = reshape(data,[],60000);
    z = bsxfun(@minus, z, mean(z,1));
    n = std(z,0,1);
    z = bsxfun(@times, z, mean(n) ./ max(n, 40));
    data = reshape(z, 32, 32, 3, []);
  end

  if opts.whitenData
    z = reshape(data,[],60000);
    W = z(:,set == 1)*z(:,set == 1)'/60000;
    [V,D] = eig(W);
    % the scale is selected to approximately preserve the norm of W
    d2 = diag(D);
    en = sqrt(mean(d2));
    z = V*diag(en./max(sqrt(d2), 10))*V'*z;
    data = reshape(z, 32, 32, 3, []);
  end

  clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

  imdb.images.data = data;
  imdb.images.labels = single(cat(2, labels{:}));
  imdb.images.set = set;
  imdb.meta.sets = {'train', 'val', 'test'};
  imdb.meta.classes = clNames.label_names;

% --------------------------------------------------------------------
function imdb = getMnistImdb(opts)
% --------------------------------------------------------------------
  % Prepare the imdb structure, returns image data with mean image subtracted
  files = {'train-images-idx3-ubyte', ...
           'train-labels-idx1-ubyte', ...
           't10k-images-idx3-ubyte', ...
           't10k-labels-idx1-ubyte'};

  if ~exist(opts.dataDir, 'dir')
    mkdir(opts.dataDir);
  end

  for i=1:4
    if ~exist(fullfile(opts.dataDir, files{i}), 'file')
      url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i});
      fprintf('downloading %s\n', url);
      gunzip(url, opts.dataDir);
    end
  end

  f=fopen(fullfile(opts.dataDir, 'train-images-idx3-ubyte'),'r');
  x1=fread(f,inf,'uint8');
  fclose(f);
  x1=permute(reshape(x1(17:end),28,28,60e3),[2 1 3]);

  f=fopen(fullfile(opts.dataDir, 't10k-images-idx3-ubyte'),'r');
  x2=fread(f,inf,'uint8');
  fclose(f);
  x2=permute(reshape(x2(17:end),28,28,10e3),[2 1 3]);

  f=fopen(fullfile(opts.dataDir, 'train-labels-idx1-ubyte'),'r');
  y1=fread(f,inf,'uint8');
  fclose(f);
  y1=double(y1(9:end)')+1;

  f=fopen(fullfile(opts.dataDir, 't10k-labels-idx1-ubyte'),'r');
  y2=fread(f,inf,'uint8');
  fclose(f);
  y2=double(y2(9:end)')+1;

  set = [ones(1,numel(y1)) 3*ones(1,numel(y2))];
  data = single(reshape(cat(3, x1, x2),28,28,1,[]));
  dataMean = mean(data(:,:,:,set == 1), 4);
  data = bsxfun(@minus, data, dataMean);

  imdb.images.data = data;
  imdb.images.data_mean = dataMean;
  imdb.images.labels = cat(2, y1, y2);
  imdb.images.set = set;
  imdb.meta.sets = {'train', 'val', 'test'};
  imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false);
