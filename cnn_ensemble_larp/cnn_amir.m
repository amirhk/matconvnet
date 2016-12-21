function [net, info] = cnn_amir(inputs_opts)
  run(fullfile(fileparts(mfilename('fullpath')), ...
    '..', 'matlab', 'vl_setupnn.m'));

  % -------------------------------------------------------------------------
  %                                                              opts.general
  % -------------------------------------------------------------------------
  opts.general.dataset = getValueFromFieldOrDefault(inputs_opts, 'dataset', 'cifar');
  opts.general.networkArch = getValueFromFieldOrDefault(inputs_opts, 'networkArch', 'lenet');
  opts.general.debugFlag = getValueFromFieldOrDefault(inputs_opts, 'debugFlag', true);

  % -------------------------------------------------------------------------
  %                                                                  opts.net
  % -------------------------------------------------------------------------
  opts.net.net = getValueFromFieldOrDefault(inputs_opts, 'network', struct()); % may optionally pass in the network
  opts.net.weightInitSource = getValueFromFieldOrDefault(inputs_opts, 'weightInitSource', 'gen');
  opts.net.weightInitSequence = getValueFromFieldOrDefault(opts, 'weightInitSequence', {'compRand', 'compRand', 'compRand'});

  % -------------------------------------------------------------------------
  %                                                                 opts.imdb
  % -------------------------------------------------------------------------
  opts.imdb.imdb = getValueFromFieldOrDefault(inputs_opts, 'imdb', struct()); % may optionally pass in the imdb
  opts.imdb.dataDir = fullfile(getDevPath(), 'data', 'source', sprintf('%s', opts.general.dataset));
  opts.imdb.whitenData = getValueFromFieldOrDefault(inputs_opts, 'whitenData', true);
  opts.imdb.contrastNormalization = getValueFromFieldOrDefault(inputs_opts, 'contrastNormalization', true);
  opts.imdb.regen = getValueFromFieldOrDefault(inputs_opts, 'regen', false);
  opts.imdb.portion = getValueFromFieldOrDefault(inputs_opts, 'portion', 1.0);

  % -------------------------------------------------------------------------
  %                                                                opts.train
  % -------------------------------------------------------------------------
  opts.train.gpus = getValueFromFieldOrDefault(inputs_opts, 'gpus', getDefaultProcessor());
  opts.train.backpropDepth = getValueFromFieldOrDefault(inputs_opts, 'backpropDepth', 4);
  opts.train.batchSize = getValueFromFieldOrDefault(inputs_opts, 'batchSize', 100);
  opts.train.errorFunction = getErrorFunctionForDataset(opts.general.dataset);
  opts.train.learningRate = getValueFromFieldOrDefault(inputs_opts, 'learningRate', [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20)]);
  opts.train.numEpochs = getValueFromFieldOrDefault(inputs_opts, 'numEpochs', numel(opts.train.learningRate));
  opts.train.weightDecay = getValueFromFieldOrDefault(inputs_opts, 'weightDecay', 0.0001);

  % -------------------------------------------------------------------------
  %                                                                opts.other
  % -------------------------------------------------------------------------
  opts.other.backpropDepthString = sprintf('bpd-%02d', opts.train.backpropDepth);
  opts.other.timeString = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.other.processorString = getProcessorStringFromProcessorList(opts.train.gpus);

  % -------------------------------------------------------------------------
  %                                                                opts.paths
  % -------------------------------------------------------------------------
  opts.paths.experimentParentDir = getValueFromFieldOrDefault( ...
    opts, ...
    'experimentParentDir', ...
    fullfile(vl_rootnn, 'experiment_results'));
  opts.paths.experimentDir = fullfile(opts.paths.experimentParentDir, sprintf( ...
    '%s-%s-%s-%s-%s', ...
    opts.general.dataset, ...
    opts.general.networkArch, ...
    opts.other.timeString, ...
    opts.other.processorString, ...
    opts.other.backpropDepthString));
  opts.paths.imdbDir = fullfile(getDevPath(), 'data', 'imdb', sprintf( ...
    '%s-%s', ...
    opts.general.dataset, ...
    opts.general.networkArch));
  opts.paths.imdbPath = fullfile(opts.paths.imdbDir, 'imdb.mat');
  opts.paths.optionsPath = fullfile(opts.paths.experimentDir, 'options.txt');
  opts.paths.resultsPath = fullfile(opts.paths.experimentDir, 'results.txt');

  % create dirs if not exist
  if ~exist(opts.paths.experimentDir)
    mkdir(opts.paths.experimentDir);
  end
  if ~exist(opts.paths.imdbDir)
    mkdir(opts.paths.imdbDir);
  end

  % -------------------------------------------------------------------------
  %                                                               get network
  % -------------------------------------------------------------------------
  output_opts = cnn_amir_init(opts);
  opts.net = mergeStructs(opts.net, output_opts.net);
  opts.train = mergeStructs(opts.train, output_opts.train);
  opts.train.weightInitSequence = printWeightInitSequence(opts.net.weightInitSequence); % TODO really needed?

  % -------------------------------------------------------------------------
  %                                                                  get imdb
  % -------------------------------------------------------------------------
  if numel(fields(opts.imdb.imdb)) % meaning an imdb was passed in as input
    imdb = opts.imdb.imdb;
  else
    if ~opts.imdb.regen && exist(opts.paths.imdbPath, 'file') % if already created, and we're not asking for regen
      imdb = load(opts.paths.imdbPath);
    else % else just construct the imdb
      switch opts.general.dataset
        case 'prostate'
          imdb = constructProstateImdb(opts);
        case 'cifar'
          imdb = constructCifarImdb(opts);
        case 'coil-100'
          imdb = constructCOIL100Imdb(opts);
        case 'mnist'
          imdb = constructMnistImdb(opts);
        case 'mnist-two-class-unbalanced'
          imdb = constructMnistUnbalancedTwoClassImdb(opts.imdb.dataDir, opts.general.networkArch);
        case 'stl-10'
          imdb = constructSTL10Imdb(opts);
      end
      if opts.general.debugFlag; afprintf(sprintf('[INFO] saving new imdb... ')); end;
      save(opts.paths.imdbPath, '-struct', 'imdb');
      if opts.general.debugFlag; afprintf(sprintf('done.\n\n')); end;
    end
  end
  opts.imdb.imdb = imdb;

  % -------------------------------------------------------------------------
  %                                   save options (don't save imdb or net!!)
  % -------------------------------------------------------------------------
  opts_copy = opts;
  % opts_copy.net = rmfield(opts_copy.net, 'net');
  opts_copy.net.net = '< too large to print net >';
  % opts_copy.imdb = rmfield(opts_copy.imdb, 'imdb');
  opts_copy.imdb.imdb = '< too large to print imdb >';
  saveStruct2File(opts_copy, opts.paths.optionsPath, 0);

  % -------------------------------------------------------------------------
  %                                                                     train
  % -------------------------------------------------------------------------
  [net, info] = cnn_train(opts.net.net, opts.imdb.imdb, getBatch(), ...
    opts.train, ...
    'experimentDir', opts.paths.experimentDir, ...
    'debugFlag', opts.general.debugFlag, ...
    'val', find(opts.imdb.imdb.images.set == 3));

  % -------------------------------------------------------------------------
  %                                             delete all but last net files
  % -------------------------------------------------------------------------
  for epoch = 1:opts.train.numEpochs - 1
    fileName = sprintf('net-epoch-%d.mat', epoch);
    delete(fullfile(opts.paths.experimentDir, fileName));
  end

  % -------------------------------------------------------------------------
  %                                        accuracy, sensitivity, specificity
  % -------------------------------------------------------------------------
  % TODO: should net & imdb even be part of the opts file?? no!
  results = {};
  predictions_train = getPredictionsFromNetOnImdb(net, imdb, 1);
  predictions_test = getPredictionsFromNetOnImdb(net, imdb, 3);
  labels_train = imdb.images.labels(imdb.images.set == 1);
  labels_test = imdb.images.labels(imdb.images.set == 3);
  [ ...
    results.train.acc, ...
    results.train.sens, ...
    results.train.spec, ...
  ] = getAccSensSpec(labels_train, predictions_train, true);
  [ ...
    results.test.acc, ...
    results.test.sens, ...
    results.test.spec, ...
  ] = getAccSensSpec(labels_test, predictions_test, true);
  saveStruct2File(results, opts.paths.resultsPath, 0);

% -------------------------------------------------------------------------
function error_function = getErrorFunctionForDataset(dataset)
% -------------------------------------------------------------------------
  if strcmp(dataset, 'prostate')
    error_function = 'multiclass-prostate';
  else
    error_function = 'multiclass';
  end

% -------------------------------------------------------------------------
function processor = getDefaultProcessor()
% -------------------------------------------------------------------------
  if ispc
    processor = [1]; % GPU at index 1
  else
    processor = [];
  end

% -------------------------------------------------------------------------
function processor_string = getProcessorStringFromProcessorList(processor_list)
% -------------------------------------------------------------------------
  if numel(processor_list)
    processor_string = sprintf('GPU-%d', processor_list(1));
  else
    processor_string = 'CPU';
  end

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
