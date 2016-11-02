function [net, info] = cnn_amir(varargin)
  run(fullfile(fileparts(mfilename('fullpath')), ...
    '..', 'matlab', 'vl_setupnn.m'));

  % Setup -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
  opts.train = struct();
  opts.networkArch = 'alex-net';
  opts.dataset = 'cifar';
  opts.imdbPortion = 1.0;
  opts.backpropDepth = 20;
  opts.weightDecay = 0.0001;
  opts.weightInitType = '1D';
  opts.weightInitSource = 'load';
  opts.bottleNeckDivideBy = 1;
  [opts, varargin] = vl_argparse(opts, varargin);
  fprintf('[INFO] networkArch:\t %s\n', opts.networkArch);
  fprintf('[INFO] dataset:\t\t %s\n', opts.dataset);
  fprintf('[INFO] imdbPortion:\t %6.5f\n', opts.imdbPortion);
  fprintf('[INFO] backpropDepth:\t %d\n', opts.backpropDepth);
  fprintf('[INFO] weightDecay:\t %6.5f\n', opts.weightDecay);
  fprintf('[INFO] weightInitType:\t %s\n', opts.weightInitType);
  fprintf('[INFO] weightInitSource: %s\n', opts.weightInitSource);
  fprintf('[INFO] bottleNeckDivideBy: %d\n', opts.bottleNeckDivideBy);
  fprintf('\n');

  % Processor -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  [opts.train.gpus, opts.processorString] = getProcessor(opts);
  [opts, varargin] = vl_argparse(opts, varargin);

  % Paths -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
  opts.timeString = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.imdbDir = fullfile(vl_rootnn, 'data', sprintf( ...
    '%s-%s', ...
    opts.dataset, ...
    opts.networkArch));
  opts.expDir = fullfile(vl_rootnn, 'data', sprintf( ...
    '%s-%s-%s-%s', ...
    opts.dataset, ...
    opts.networkArch, ...
    opts.timeString, ...
    opts.processorString));
  [opts, varargin] = vl_argparse(opts, varargin);
  if ~exist(opts.expDir)
    mkdir(opts.expDir);
  end
  opts.dataDir = fullfile(vl_rootnn, 'data', opts.dataset);
  opts.imdbPath = fullfile(opts.imdbDir, 'imdb.mat');
  if ~exist(opts.imdbDir)
    mkdir(opts.imdbDir);
  else
    % if folder exists, there may be an imdb inside there (that corresponds to
    % a different portion of CIFAR). just delete the imdb and remake to be safe.
    delete(fullfile(opts.imdbDir, 'imdb.mat'));
  end

  % IMDB -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  opts.whitenData = true;
  opts.contrastNormalization = true;
  opts = vl_argparse(opts, varargin);

  % -------------------------------------------------------------------------
  %                                                    Prepare model and data
  % -------------------------------------------------------------------------
  net = cnn_amir_init( ...
    'networkArch', opts.networkArch, ...
    'dataset', opts.dataset, ...
    'backpropDepth', opts.backpropDepth, ...
    'weightDecay', opts.weightDecay, ...
    'weightInitType', opts.weightInitType, ...
    'weightInitSource', opts.weightInitSource, ...
    'bottleNeckDivideBy', opts.bottleNeckDivideBy);
  saveNetworkInfo(net, opts.expDir);

  if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath);
  else
    imdb = constructCifarImdb(opts);
    fprintf('[INFO] saving new imdb... ');
    save(opts.imdbPath, '-struct', 'imdb');
    fprintf('done.\n\n');
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
      freeGPUIndex = getFreeGPUIndex();
      % freeGPUIndex = 1;
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
function imdb = constructCifarImdb(opts)
% -------------------------------------------------------------------------
  fprintf('[INFO] Constructing CIFAR imdb (portion = %%%d)...\n\n', opts.imdbPortion * 100);
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

  data = single(cat(4, data{:}));
  labels = single(cat(2, labels{:}));
  set = cat(2, sets{:});

  % remove mean in any case
  dataMean = mean(data(:,:,:,set == 1), 4);
  data = bsxfun(@minus, data, dataMean);

  [output_data, output_labels] = choosePortionOfImdb(data(:,:,:,1:50000), labels(1:50000), opts.imdbPortion);
  data = single(cat(4, output_data, data(:,:,:,50001:60000))); % amend with test data
  labels = single(cat(2, output_labels, labels(50001:60000))); % amend with test data
  set = [ones(1, 50000 * opts.imdbPortion) 3 * ones(1, 10000)]; % all of the test portion
  number_of_train_and_test_images = size(labels, 2);
  fprintf('[INFO] number_of_train_and_test_images in portion: %d.\n', number_of_train_and_test_images);

  % normalize by image mean and std as suggested in `An Analysis of
  % Single-Layer Networks in Unsupervised Feature Learning` Adam
  % Coates, Honglak Lee, Andrew Y. Ng

  if opts.contrastNormalization
    fprintf('[INFO] contrast-normalizing data... ');
    z = reshape(data,[],number_of_train_and_test_images);
    z = bsxfun(@minus, z, mean(z,1));
    n = std(z,0,1);
    z = bsxfun(@times, z, mean(n) ./ max(n, 40));
    data = reshape(z, 32, 32, 3, []);
    fprintf('done.\n');
  end

  if opts.whitenData
    fprintf('[INFO] whitening data... ');
    z = reshape(data,[],number_of_train_and_test_images);
    W = z(:,set == 1)*z(:,set == 1)'/number_of_train_and_test_images;
    [V,D] = eig(W);
    % the scale is selected to approximately preserve the norm of W
    d2 = diag(D);
    en = sqrt(mean(d2));
    z = V*diag(en./max(sqrt(d2), 10))*V'*z;
    data = reshape(z, 32, 32, 3, []);
    fprintf('done.\n');
  end

  clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

  imdb.images.data = data;
  imdb.images.labels = labels;
  imdb.images.set = set;
  imdb.meta.sets = {'train', 'val', 'test'};
  imdb.meta.classes = clNames.label_names;
  fprintf('[INFO] Finished constructing CIFAR imdb (portion = %%%d)!\n', opts.imdbPortion * 100);

% -------------------------------------------------------------------------
function [data, labels] = choosePortionOfImdb(data, labels, portion)
% -------------------------------------------------------------------------
  % VERY INEFFICIENT
  number_of_classes = 10;
  number_of_samples = size(labels, 2);
  number_of_images_per_class = number_of_samples / number_of_classes * portion;

  label_indices = {};
  output_data = {};
  output_labels = {};
  for i = 1:number_of_classes
    label_indices{i} = (labels == i);
    fprintf('\t[INFO] found %d images with label %d...\n', size(label_indices{i}, 2), i);
  end
  fprintf('\n');

  tic;
  for i = 1:number_of_classes
    fprintf('\t[INFO] extracting images for class %d...', i);
    output_data{i} = data(:,:,:,label_indices{i});
    output_labels{i} = labels(label_indices{i});
    fprintf('done! \t');
    toc;
  end
  fprintf('\n');

  for i = 1:number_of_classes
    portioned_output_data{i} = output_data{i}(:,:,:,1:number_of_images_per_class);
    portioned_output_labels{i} = output_labels{i}(1:number_of_images_per_class);
  end

  data = single(cat(4, output_data{:}));
  labels = single(cat(2, output_labels{:}));

  % shuffle data and labels the same way
  ix = randperm(number_of_samples * portion);
  data = data(:,:,:,ix);
  labels = labels(ix);
