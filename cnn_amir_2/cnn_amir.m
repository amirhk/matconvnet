function [net, info] = cnn_amir(varargin)
  run(fullfile(fileparts(mfilename('fullpath')), ...
    '..', 'matlab', 'vl_setupnn.m'));

  % Setup -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
  opts.train = struct();
  opts.folderNumber = 2;
  opts.networkArch = 'alexnet';
  opts.dataset = 'cifar';
  opts.imdbPortion = 1.0;
  opts.backpropDepth = 20;
  opts.weightDecay = 0.0001;
  opts.weightInitSequence = {'1D', 'compRand', '1D', '2D-shiftflip', '1D'};
  opts.weightInitSource = 'load';
  opts.bottleneckDivideBy = 1;
  [opts, varargin] = vl_argparse(opts, varargin);
  fprintf('[INFO] networkArch:\t %s\n', opts.networkArch);
  fprintf('[INFO] dataset:\t\t %s\n', opts.dataset);
  fprintf('[INFO] imdbPortion:\t %6.5f\n', opts.imdbPortion);
  fprintf('[INFO] backpropDepth:\t %d\n', opts.backpropDepth);
  fprintf('[INFO] weightDecay:\t %6.5f\n', opts.weightDecay);
  fprintf('[INFO] weightInitSequence:\t %s\n', printWeightInitSequence(opts.weightInitSequence));
  fprintf('[INFO] weightInitSource: %s\n', opts.weightInitSource);
  fprintf('[INFO] bottleneckDivideBy: %d\n', opts.bottleneckDivideBy);
  fprintf('\n');

  % Processor -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  [opts.train.gpus, opts.processorString] = getProcessor(opts);
  [opts, varargin] = vl_argparse(opts, varargin);

  % Paths -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
  opts.timeString = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.dataFolderString = sprintf('data_%d', opts.folderNumber);
  opts.dataDir = fullfile(vl_rootnn, opts.dataFolderString, sprintf('_%s', opts.dataset));
  opts.imdbDir = fullfile(vl_rootnn, opts.dataFolderString, sprintf( ...
    '%s-%s', ...
    opts.dataset, ...
    opts.networkArch));
  opts.imdbPath = fullfile(opts.imdbDir, 'imdb.mat');
  if ~exist(opts.imdbDir)
    mkdir(opts.imdbDir);
  else
    % if folder exists, there may be an imdb inside there (that corresponds to
    % a different portion of CIFAR). just delete the imdb and remake to be safe.
    if opts.imdbPortion ~= 1
      delete(fullfile(opts.imdbDir, 'imdb.mat'));
    end
  end
  opts.expDir = fullfile(vl_rootnn, opts.dataFolderString, sprintf( ...
    '%s-%s-%s-%s', ...
    opts.dataset, ...
    opts.networkArch, ...
    opts.timeString, ...
    opts.processorString));
  [opts, varargin] = vl_argparse(opts, varargin);
  if ~exist(opts.expDir)
    mkdir(opts.expDir);
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
    'weightInitSequence', opts.weightInitSequence, ...
    'weightInitSource', opts.weightInitSource, ...
    'bottleneckDivideBy', opts.bottleneckDivideBy);
  saveNetworkInfo(net, opts.expDir);

  if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath);
  else
    switch opts.dataset
      case 'mnist'
        imdb = constructMnistImdb(opts);
      case 'cifar'
        imdb = constructCifarImdb(opts);
      case 'stl-10'
        imdb = constructSTL10Imdb(opts);
    end
    fprintf('[INFO] saving new imdb... ');
    save(opts.imdbPath, '-struct', 'imdb');
    fprintf('done.\n\n');
  end

  % net.meta.classes.name = imdb.meta.classes(:)';

  % -------------------------------------------------------------------------
  %                                                                     Train
  % -------------------------------------------------------------------------
  [net, info] = cnn_train(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, ...
    net.meta.trainOpts, ...
    opts.train, ...
    'val', find(imdb.images.set == 3));

  % -------------------------------------------------------------------------
  %                                             Delete All But Last Net Files
  % -------------------------------------------------------------------------
  for epoch = 1:net.meta.trainOpts.numEpochs - 1
    fileName = sprintf('net-epoch-%d.mat', epoch);
    delete(fullfile(opts.expDir, fileName));
  end

% -------------------------------------------------------------------------
function [processorList, processorString] = getProcessor(opts)
% -------------------------------------------------------------------------
  if ~isfield(opts.train, 'gpus')
    if ispc
      % freeGPUIndex = getFreeGPUIndex();
      % freeGPUIndex = 1;
      freeGPUIndex = opts.folderNumber;
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
