function [net, info] = cnn_amir(varargin)
  run(fullfile(fileparts(mfilename('fullpath')), ...
    '..', 'matlab', 'vl_setupnn.m'));

  fileName = mfilename; % 'cnn_amir'
  fullFilePath = mfilename('fullpath'); % '/Users/a6karimi/dev/matconvnet/cnn_amir_1/cnn_amir'
  parentFolderPath = fullFilePath(1:end-length(fileName)-1);
  folderNumber = str2num(parentFolderPath(end:end));

  % Setup -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
  opts.imdb = struct();
  opts.train = struct();
  opts.imdbOptions = {};
  opts.leaveOutType = 'sample';
  opts.leaveOutIndex = 1;
  opts.folderNumber = 2; % TODO: change!!!!
  opts.networkArch = 'alexnet';
  opts.dataset = 'cifar';
  opts.regenDatabase = 0;
  opts.backpropDepth = 20;
  opts.weightDecay = 0.0001;
  opts.weightInitSequence = {'1D', 'compRand', '1D', '2D-shiftflip', '1D'};
  opts.weightInitSource = 'load';
  opts.bottleneckDivideBy = 1;
  opts.debugFlag = true;
  [opts, varargin] = vl_argparse(opts, varargin);
  if opts.debugFlag
    fprintf('\n-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --\n\n');
    fprintf('[INFO] networkArch:\t %s\n', opts.networkArch);
    fprintf('[INFO] dataset:\t\t %s\n', opts.dataset);
    fprintf('[INFO] backpropDepth:\t %d\n', opts.backpropDepth);
    fprintf('[INFO] weightDecay:\t %6.5f\n', opts.weightDecay);
    fprintf('[INFO] weightInitSequence:\t %s\n', printWeightInitSequence(opts.weightInitSequence));
    fprintf('[INFO] weightInitSource: %s\n', opts.weightInitSource);
    fprintf('[INFO] bottleneckDivideBy: %d\n', opts.bottleneckDivideBy);
    fprintf('\n');
  end

  % Processor -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  [opts.train.gpus, opts.processorString] = getProcessor(opts);
  [opts, varargin] = vl_argparse(opts, varargin);

  % Paths -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
  opts.timeString = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.backpropDepthString = sprintf('bpd-%02d', opts.backpropDepth);
  opts.dataFolderString = sprintf('data_%d', opts.folderNumber);
  opts.dataDir = fullfile(vl_rootnn, opts.dataFolderString, sprintf('_%s', opts.dataset));
  opts.imdbDir = fullfile(vl_rootnn, opts.dataFolderString, sprintf( ...
    '%s-%s', ...
    opts.dataset, ...
    opts.networkArch));
  opts.imdbPath = fullfile(opts.imdbDir, 'imdb.mat');
  %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
  %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
  opts.imdbBalancedDir = fullfile(vl_rootnn, opts.dataFolderString, sprintf( ...
    'balanced-%s-%s', ...
    opts.dataset, ...
    opts.networkArch));
  opts.imdbBalancedPath = fullfile(opts.imdbBalancedDir, 'imdb.mat');
  %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
  %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
  if ~exist(opts.imdbDir)
    mkdir(opts.imdbDir);
  else
    % if folder exists, there may be an imdb inside there (that corresponds to
    % a different portion of CIFAR). just delete the imdb and remake to be safe.
    if opts.regenDatabase
      delete(fullfile(opts.imdbDir, 'imdb.mat'));
    end
  end
  opts.expDir = fullfile(vl_rootnn, opts.dataFolderString, sprintf( ...
    '%s-%s-%s-%s-%s', ...
    opts.dataset, ...
    opts.networkArch, ...
    opts.timeString, ...
    opts.processorString, ...
    opts.backpropDepthString));
  [opts, varargin] = vl_argparse(opts, varargin);
  if ~exist(opts.expDir)
    mkdir(opts.expDir);
  end

  % IMDB -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  opts.whitenData = true;
  opts.contrastNormalization = true;
  [opts, varargin] = vl_argparse(opts, varargin);

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
  saveNetworkInfo(net, opts);

  % TODO: make this better amir!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  if numel(varargin) == 2 && strcmp(varargin{1}, 'imdb')
    imdb = varargin{2};
  elseif exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath);
  else
    switch opts.dataset
      case 'prostate'
        imdb = constructProstateImdb(opts);
      case 'cifar'
        imdb = constructCifarImdb(opts);
      case 'coil-100'
        imdb = constructCOIL100Imdb(opts);
      case 'mnist'
        imdb = constructMnistImdb(opts);
      case 'stl-10'
        imdb = constructSTL10Imdb(opts);
    end
    if opts.debugFlag; fprintf('[INFO] saving new imdb... '); end;
    save(opts.imdbPath, '-struct', 'imdb');
    if opts.debugFlag; fprintf('done.\n\n'); end;
  end

  if strcmp(opts.dataset, 'prostate')
    opts.errorFunction = 'multiclass-prostate';
  else
    opts.errorFunction = 'multiclass';
  end

  % net.meta.classes.name = imdb.meta.classes(:)';

  % -------------------------------------------------------------------------
  %                                                                     Train
  % -------------------------------------------------------------------------
  [net, info] = cnn_train(net, imdb, getBatch(), ...
    'expDir', opts.expDir, ...
    'errorFunction', opts.errorFunction, ...
    'debugFlag', opts.debugFlag, ...
    net.meta.trainOpts, ...
    opts.train, ...
    'val', find(imdb.images.set == 3));

  saveFinalSensitivitySpecificityInfo(info, opts.expDir);

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
function saveNetworkInfo(net, opts)
% -------------------------------------------------------------------------
  if opts.debugFlag; fprintf('\n[INFO] Saving network info in readme... '); end;
  struct2File( ...
    net.meta.trainOpts, ...
    fullfile(opts.expDir, 'readme.txt'), ...
    'delimiter', ...
    '\n\n');
  if opts.debugFlag; fprintf('done!\n'); end;


% -------------------------------------------------------------------------
function saveFinalSensitivitySpecificityInfo(info, expDir)
% -------------------------------------------------------------------------
  fileID = fopen(fullfile(expDir, 'sensitivity_specificity.txt'), 'w');
  fprintf(fileID, '-- -- -- -- -- -- FINAL TRAINING VALUES -- -- -- -- -- --\n');
  fprintf(fileID, '\t[INFO] Accuracy: %6.5f\n', 1 - info.train.error(end));
  fprintf(fileID, '\t[INFO] TP: %d\n', info.train.stats.TP);
  fprintf(fileID, '\t[INFO] TN: %d\n', info.train.stats.TN);
  fprintf(fileID, '\t[INFO] FP: %d\n', info.train.stats.FP);
  fprintf(fileID, '\t[INFO] FN: %d\n', info.train.stats.FN);
  fprintf(fileID, '\t[INFO] Sensitivity: %6.5f\n', info.train.stats.sensitivity);
  fprintf(fileID, '\t[INFO] Specificity: %6.5f\n', info.train.stats.specificity);
  fprintf(fileID, '\t\n\n');
  fprintf(fileID, '-- -- -- -- -- -- FINAL TESTING VALUES -- -- -- -- -- --\n');
  fprintf(fileID, '\t[INFO] Accuracy: %6.5f\n', 1 - info.val.error(end));
  fprintf(fileID, '\t[INFO] TP: %d\n', info.val.stats.TP);
  fprintf(fileID, '\t[INFO] TN: %d\n', info.val.stats.TN);
  fprintf(fileID, '\t[INFO] FP: %d\n', info.val.stats.FP);
  fprintf(fileID, '\t[INFO] FN: %d\n', info.val.stats.FN);
  fprintf(fileID, '\t[INFO] Sensitivity: %6.5f\n', info.val.stats.sensitivity);
  fprintf(fileID, '\t[INFO] Specificity: %6.5f\n', info.val.stats.specificity);
  fprintf(fileID, '\t\n\n');
  fclose(fileID);


% -------------------------------------------------------------------------
function fn = getBatch()
% -------------------------------------------------------------------------
  fn = @(x,y) getSimpleNNBatch(x,y);

% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
  images = imdb.images.data(:,:,:,batch);
  labels = imdb.images.labels(1,batch);
  if rand > 0.5, images=fliplr(images); end
