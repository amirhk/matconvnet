% -------------------------------------------------------------------------
function [trained_model, performance_summary] = testCnn(input_opts)
% -------------------------------------------------------------------------
% Copyright (c) 2017, Amir-Hossein Karimi
% All rights reserved.

% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution

% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.

  run(fullfile(fileparts(mfilename('fullpath')), ...
    '..', 'matlab', 'vl_setupnn.m'));

  % -------------------------------------------------------------------------
  %                                                              opts.general
  % -------------------------------------------------------------------------
  opts.general.dataset = getValueFromFieldOrDefault(input_opts, 'dataset', 'cifar');
  opts.general.network_arch = getValueFromFieldOrDefault(input_opts, 'network_arch', 'lenet');
  opts.general.return_performance_summary = getValueFromFieldOrDefault(input_opts, 'return_performance_summary', true);
  opts.general.debug_flag = getValueFromFieldOrDefault(input_opts, 'debug_flag', false);

  % -------------------------------------------------------------------------
  %                                                                  opts.net
  % -------------------------------------------------------------------------
  opts.net.net = getValueFromFieldOrDefault(input_opts, 'net', struct()); % may optionally pass in the network
  opts.net.weight_init_source = getValueFromFieldOrDefault(input_opts, 'weight_init_source', 'gen');
  opts.net.weight_init_sequence = getValueFromFieldOrDefault(input_opts, 'weight_init_sequence', {'compRand', 'compRand', 'compRand', 'compRand', 'compRand'});
  opts.net.bottleneck_structure = getValueFromFieldOrDefault(input_opts, 'bottleneck_structure', []);

  % -------------------------------------------------------------------------
  %                                                                 opts.imdb
  % -------------------------------------------------------------------------
  opts.imdb.imdb = getValueFromFieldOrDefault(input_opts, 'imdb', struct()); % may optionally pass in the imdb
  opts.imdb.data_dir = fullfile(getDevPath(), 'data', 'source', sprintf('%s', opts.general.dataset));
  opts.imdb.whiten_data = getValueFromFieldOrDefault(input_opts, 'whiten_data', true);
  opts.imdb.contrast_normalization = getValueFromFieldOrDefault(input_opts, 'contrast_normalization', true);
  opts.imdb.regen = getValueFromFieldOrDefault(input_opts, 'regen', false);
  opts.imdb.portion = getValueFromFieldOrDefault(input_opts, 'portion', 1.0);

  % -------------------------------------------------------------------------
  %                                                                opts.train
  % -------------------------------------------------------------------------
  opts.train.gpus = ifNotMacSetGpu(getValueFromFieldOrDefault(input_opts, 'gpus', 1));
  opts.train.backprop_depth = getValueFromFieldOrDefault(input_opts, 'backprop_depth', 4);
  opts.train.batch_size = getValueFromFieldOrDefault(input_opts, 'batch_size', 100);
  opts.train.error_function = getErrorFunctionForDataset(opts.general.dataset);
  opts.train.learning_rate = getValueFromFieldOrDefault(input_opts, 'learning_rate', 'default_keyword');
  opts.train.num_epochs = getValueFromFieldOrDefault(input_opts, 'num_epochs', numel(opts.train.learning_rate));
  opts.train.weight_decay = getValueFromFieldOrDefault(input_opts, 'weight_decay', 0.0001);

  % -------------------------------------------------------------------------
  %                                                                opts.other
  % -------------------------------------------------------------------------
  opts.other.backprop_depth_string = sprintf('bpd-%02d', opts.train.backprop_depth);
  opts.other.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.other.processor_string = getProcessorStringFromProcessorList(opts.train.gpus);

  % -------------------------------------------------------------------------
  %                                                                opts.paths
  % -------------------------------------------------------------------------
  opts.paths.experiment_parent_dir = getValueFromFieldOrDefault( ...
    input_opts, ...
    'experiment_parent_dir', ...
    fullfile(vl_rootnn, 'experiment_results'));
  opts.paths.experiment_dir = fullfile(opts.paths.experiment_parent_dir, sprintf( ...
    'cnn-%s-%s-%s-%s-%s', ...
    opts.other.time_string, ...
    opts.general.dataset, ...
    opts.general.network_arch, ...
    opts.other.processor_string, ...
    opts.other.backprop_depth_string));
  opts.paths.imdb_dir = fullfile(getDevPath(), 'data', 'imdb', sprintf( ...
    '%s-%s', ...
    opts.general.dataset, ...
    opts.general.network_arch));
  opts.paths.imdb_path = fullfile(opts.paths.imdb_dir, 'imdb.mat');
  opts.paths.options_file_path = fullfile(opts.paths.experiment_dir, 'options.txt');
  opts.paths.results_file_path = fullfile(opts.paths.experiment_dir, 'results.txt');

  % create dirs if not exist
  if ~exist(opts.paths.experiment_dir)
    mkdir(opts.paths.experiment_dir);
  end
  if ~exist(opts.paths.imdb_dir)
    mkdir(opts.paths.imdb_dir);
  end

  % -------------------------------------------------------------------------
  %                                                               get network
  % -------------------------------------------------------------------------
  if numel(opts.net.bottleneck_structure) > 0
    network_opts = cnnInitWithBottlenecks(opts);
  else
    network_opts = cnnInit(opts);
  end
  if ~length(fieldnames(opts.net.net))
    % opts.net = mergeStructs(opts.net, network_opts.net);
    opts.net.net = network_opts.net;
  else
    opts.net.net = opts.net.net; % use the network passed in this function
  end
  opts.train = mergeStructs(opts.train, network_opts.train);
  % opts.train.weight_init_sequence = printWeightInitSequence(opts.net.weight_init_sequence); % TODO really needed?

  % -------------------------------------------------------------------------
  %                                                                  get imdb
  % -------------------------------------------------------------------------
  if numel(fields(opts.imdb.imdb)) % meaning an imdb was passed in as input
    imdb = opts.imdb.imdb;
  else
    if exist(opts.paths.imdb_path, 'file') && ~opts.imdb.regen % if already created, and we're not asking for regen
      imdb = load(opts.paths.imdb_path);
    else % else just construct the imdb
      switch opts.general.dataset
        % TODO: remove completely.... just always load a saved imdb... no regen unless manually supervised by amir
        % case '*-two-class'
          % unbalanced imdbs should always be generated separately and consistent across all tests
        case 'cifar'
          imdb = constructCifarImdb(opts);
        case 'coil-100'
          imdb = constructCoil100Imdb(opts);
        case 'mnist'
          imdb = constructMnistImdb(opts);
        case 'prostate'
          imdb = constructProstateImdb(opts);
        case 'stl-10'
          imdb = constructStl10Imdb(opts);
        case 'svhn'
          imdb = constructSvhnImdb(opts);
      end
      if opts.general.debug_flag; afprintf(sprintf('[INFO] saving new imdb... ')); end;
      save(opts.paths.imdb_path, '-struct', 'imdb');
      if opts.general.debug_flag; afprintf(sprintf('done.\n\n')); end;
    end
  end

  opts.imdb.imdb = imdb;

  % -------------------------------------------------------------------------
  %                          save experiment setup (don't save imdb or net!!)
  % -------------------------------------------------------------------------
  opts_copy = opts;
  % opts_copy.net = rmfield(opts_copy.net, 'net');
  % opts_copy.imdb = rmfield(opts_copy.imdb, 'imdb');
  opts_copy.net.net = '< too large to print net >';
  opts_copy.imdb.imdb = '< too large to print imdb >';
  saveStruct2File(opts_copy, opts.paths.options_file_path, 0);

  % -------------------------------------------------------------------------
  %                                                                     train
  % -------------------------------------------------------------------------
  [net, info] = cnnTrain(opts.net.net, opts.imdb.imdb, getBatch(), ...
    opts.train, ...
    'experiment_dir', opts.paths.experiment_dir, ...
    'debug_flag', opts.general.debug_flag, ...
    'val', find(opts.imdb.imdb.images.set == 3));

  % -------------------------------------------------------------------------
  %                                             delete all but last net files
  % -------------------------------------------------------------------------
  for epoch = 1:opts.train.num_epochs - 1
    file_name = sprintf('net-epoch-%d.mat', epoch);
    delete(fullfile(opts.paths.experiment_dir, file_name));
  end

  % -------------------------------------------------------------------------
  %                                                   get performance summary
  % -------------------------------------------------------------------------
  if opts.general.return_performance_summary
    if isTwoClassImdb(opts.general.dataset)
      [top_train_predictions, ~] = getPredictionsFromModelOnImdb(net, 'cnn', imdb, 1);
      afprintf(sprintf('[INFO] Getting model performance on `train` set...\n'));
      labels_train = imdb.images.labels(imdb.images.set == 1);
      afprintf(sprintf('[INFO] Model performance on `train` set\n'));
      [ ...
        train_accuracy, ...
        train_sensitivity, ...
        train_specificity, ...
      ] = getAccSensSpec(labels_train, top_train_predictions, true);
      [top_test_predictions, ~] = getPredictionsFromModelOnImdb(net, 'cnn', imdb, 3);
      afprintf(sprintf('[INFO] Getting model performance on `test` set...\n'));
      afprintf(sprintf('[INFO] Model performance on `test` set\n'));
      labels_test = imdb.images.labels(imdb.images.set == 3);
      [ ...
        test_accuracy, ...
        test_sensitivity, ...
        test_specificity, ...
      ] = getAccSensSpec(labels_test, top_test_predictions, true);
      printConsoleOutputSeparator();
    else
      train_accuracy = 1 - info.train.error(1,end);
      train_sensitivity = -1;
      train_specificity = -1;
      test_accuracy = 1 - info.val.error(1,end);
      test_sensitivity = -1;
      test_specificity = -1;
    end
  else
    train_accuracy = -1;
    train_sensitivity = -1;
    train_specificity = -1;
    test_accuracy = -1;
    test_sensitivity = -1;
    test_specificity = -1;
  end

  % -------------------------------------------------------------------------
  %                                                             assign output
  % -------------------------------------------------------------------------
  trained_model = net;
  % performance_summary.info = info;
  performance_summary.train.accuracy = train_accuracy;
  performance_summary.train.sensitivity = train_sensitivity;
  performance_summary.train.specificity = train_specificity;
  performance_summary.test.accuracy = test_accuracy;
  performance_summary.test.sensitivity = test_sensitivity;
  performance_summary.test.specificity = test_specificity;

  % -------------------------------------------------------------------------
  %                                                               save output
  % -------------------------------------------------------------------------
  saveStruct2File(performance_summary, opts.paths.results_file_path, 0);

% -------------------------------------------------------------------------
function error_function = getErrorFunctionForDataset(dataset)
% -------------------------------------------------------------------------
  if isTwoClassImdb(dataset)
    error_function = 'two-class';
  else
    error_function = 'multiclass';
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
