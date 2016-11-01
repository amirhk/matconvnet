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

  % IMDB -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % opts.imdbPortion = 0.1;
  % opts.imdbPortion = 1;
  opts.imdbPortion = .25;
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
    imdb = constructCifarImdb(opts);
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
function imdb = constructCifarImdb(opts)
% -------------------------------------------------------------------------
  fprintf('[INFO] Constructing CIFAR imdb (portion = %d%%)...\n\n', opts.imdbPortion * 100);
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

  % TODO: something here to specify whether to build full or partial CIFAR
  % opts.imdbPortion;

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

  [data, labels, set] = choosePortionOfImdb(data, labels, set, opts.imdbPortion);
  % TODO remove set code....
  set = [ones(1, 50000 * opts.imdbPortion) 3 * ones(1, 10000 * opts.imdbPortion)];

  % normalize by image mean and std as suggested in `An Analysis of
  % Single-Layer Networks in Unsupervised Feature Learning` Adam
  % Coates, Honglak Lee, Andrew Y. Ng

  if opts.contrastNormalization
    % z = reshape(data,[],60000);
    z = reshape(data,[],size(labels, 2));
    z = bsxfun(@minus, z, mean(z,1));
    n = std(z,0,1);
    z = bsxfun(@times, z, mean(n) ./ max(n, 40));
    data = reshape(z, 32, 32, 3, []);
  end

  if opts.whitenData
    % z = reshape(data,[],60000);
    % W = z(:,set == 1)*z(:,set == 1)'/60000;
    z = reshape(data,[],size(labels, 2));
    W = z(:,set == 1)*z(:,set == 1)'/size(labels, 2);
    [V,D] = eig(W);
    % the scale is selected to approximately preserve the norm of W
    d2 = diag(D);
    en = sqrt(mean(d2));
    z = V*diag(en./max(sqrt(d2), 10))*V'*z;
    data = reshape(z, 32, 32, 3, []);
  end

  clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

  imdb.images.data = data;
  imdb.images.labels = labels
  imdb.images.set = set;
  imdb.meta.sets = {'train', 'val', 'test'};
  imdb.meta.classes = clNames.label_names;
  fprintf('[INFO] Finished constructing CIFAR imdb (portion = %d%%)!\n', opts.imdbPortion * 100);

% -------------------------------------------------------------------------
function [data, labels, set] = choosePortionOfImdb(data, labels, set, portion)
% -------------------------------------------------------------------------
  % VERY INEFFICIENT
  number_of_classes = 10;
  number_of_samples = size(labels, 2);

  label_indices = {};
  output_data = {};
  output_labels = {};
  output_set = {};
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
    output_set{i} = labels(label_indices{i});
    fprintf('done! \t');
    toc;
  end
  fprintf('\n');

  for i = 1:number_of_classes
    portioned_output_data{i} = output_data{i}(:,:,:,1:number_of_samples / number_of_classes * portion);
    portioned_output_labels{i} = output_labels{i}(1:number_of_samples / number_of_classes * portion);
    portioned_output_set{i} = output_set{i}(1:number_of_samples / number_of_classes * portion);
  end

  data = single(cat(4, output_data{:}));
  labels = single(cat(2, output_labels{:}));
  set = cat(2, output_set{:});

  ix = randperm(number_of_samples * portion);
  data = data(:,:,:,ix);
  labels = labels(ix);
  set = set(ix);

  % tic;
  % for i = 1:number_of_classes
  %   fprintf('\t[INFO] extracting images for class %d...', i);
  %   output_data{i} = [];
  %   output_labels{i} = [];
  %   output_set{i} = [];
  %   count = 0;
  %   for j = 1:number_of_samples
  %     if label_indices{i}(j) == 1;
  %       count = count + 1;
  %       output_data{i} = cat(4, output_data{i}, data(:,:,:,j));
  %       output_labels{i} = cat(2, output_labels{i}, labels(j));
  %       output_set{i} = cat(2, output_set{i}, labels(j));
  %     end
  %     if count == number_of_samples / number_of_classes * portion
  %       fprintf('done! \t');
  %       toc;
  %       break
  %     end
  %   end
  % end
  % fprintf('\n');

  % % disp(output_data);
  % % disp(size(output_data{1}));
  % data = single(cat(4, output_data{:}));
  % labels = single(cat(2, output_labels{:}));
  % set = cat(2, output_set{:});








  % disp(size(data));
  % disp(size(labels));
  % disp(size(set));

  % for i = 1:number_of_classes
  %   portioned_output_data{i} = output_data{i}(:,:,:,1:number_of_samples / number_of_classes * portion);
  %   portioned_output_labels{i} = output_labels{i}(1:number_of_samples / number_of_classes * portion);
  %   portioned_output_set{i} = output_set{i}(1:number_of_samples / number_of_classes * portion);
  % end

  % % finally shuffle these into 1 list for data, labels, set and pass that pack
  % % make sure to shuffle all the same way!
  % data = single(cat(4, portioned_output_data{:}));
  % labels = single(cat(2, portioned_output_labels{:}));
  % set = cat(2, portioned_output_set{:});

  % TODO: not necessary, but great to have,... shuffle the above 3 matrices
  % together in same order (not required bc cnn_train randomly chooses anyways)
