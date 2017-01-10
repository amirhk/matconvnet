function [net, results] = cnn_amir(inputs_opts)
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
  opts.general.dataset = getValueFromFieldOrDefault(inputs_opts, 'dataset', 'cifar');
  opts.general.network_arch = getValueFromFieldOrDefault(inputs_opts, 'network_arch', 'lenet');
  if strcmp(opts.general.dataset, 'prostate-v2-20-patients') || ...
    strcmp(opts.general.dataset, 'mnist-two-class-9-4') || ...
    strcmp(opts.general.dataset, 'svhn-two-class-9-4') || ...
    strcmp(opts.general.dataset, 'cifar-two-deer-horse') || ...
    strcmp(opts.general.dataset, 'cifar-two-deer-truck')
    assert(strcmp(opts.general.network_arch, 'two-class-lenet'));
  end
  opts.general.debug_flag = getValueFromFieldOrDefault(inputs_opts, 'debug_flag', true);

  % -------------------------------------------------------------------------
  %                                                                  opts.net
  % -------------------------------------------------------------------------
  opts.net.net = getValueFromFieldOrDefault(inputs_opts, 'network', struct()); % may optionally pass in the network
  opts.net.weight_init_source = getValueFromFieldOrDefault(inputs_opts, 'weight_init_source', 'gen');
  opts.net.weight_init_sequence = getValueFromFieldOrDefault(inputs_opts, 'weight_init_sequence', {'compRand', 'compRand', 'compRand'});

  % -------------------------------------------------------------------------
  %                                                                 opts.imdb
  % -------------------------------------------------------------------------
  opts.imdb.imdb = getValueFromFieldOrDefault(inputs_opts, 'imdb', struct()); % may optionally pass in the imdb
  opts.imdb.data_dir = fullfile(getDevPath(), 'data', 'source', sprintf('%s', opts.general.dataset));
  opts.imdb.whiten_data = getValueFromFieldOrDefault(inputs_opts, 'whiten_data', true);
  opts.imdb.contrast_normalization = getValueFromFieldOrDefault(inputs_opts, 'contrast_normalization', true);
  opts.imdb.regen = getValueFromFieldOrDefault(inputs_opts, 'regen', false);
  opts.imdb.portion = getValueFromFieldOrDefault(inputs_opts, 'portion', 1.0);

  % -------------------------------------------------------------------------
  %                                                                opts.train
  % -------------------------------------------------------------------------
  opts.train.gpus = getValueFromFieldOrDefault(inputs_opts, 'gpus', getDefaultProcessor());
  opts.train.backprop_depth = getValueFromFieldOrDefault(inputs_opts, 'backprop_depth', 4);
  opts.train.batch_size = getValueFromFieldOrDefault(inputs_opts, 'batch_size', 100);
  opts.train.error_function = getErrorFunctionForDataset(opts.general.dataset);
  opts.train.learning_rate = getValueFromFieldOrDefault(inputs_opts, 'learning_rate', [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20)]);
  opts.train.num_epochs = getValueFromFieldOrDefault(inputs_opts, 'num_epochs', numel(opts.train.learning_rate));
  opts.train.weight_decay = getValueFromFieldOrDefault(inputs_opts, 'weight_decay', 0.0001);

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
    inputs_opts, ...
    'experiment_parent_dir', ...
    fullfile(vl_rootnn, 'experiment_results'));
  opts.paths.experiment_dir = fullfile(opts.paths.experiment_parent_dir, sprintf( ...
    '%s-%s-%s-%s-%s', ...
    opts.general.dataset, ...
    opts.general.network_arch, ...
    opts.other.time_string, ...
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
  output_opts = cnnAmirInit(opts);
  opts.net = mergeStructs(opts.net, output_opts.net);
  opts.train = mergeStructs(opts.train, output_opts.train);
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
  %                                        accuracy, sensitivity, specificity
  % -------------------------------------------------------------------------
  % TODO: should net & imdb even be part of the opts file?? no!
  [ST,~] = dbstack();
  results = {};
  if numel(ST) >= 2 && strcmp(ST(2).file, 'mainCnnAmir.m') || strcmp(ST(2).file, 'testSingleNetwork.m')
    [top_train_predictions, all_train_predictions] = getPredictionsFromModelOnImdb(net, 'cnn', imdb, 1);
    labels_train = imdb.images.labels(imdb.images.set == 1);
    [ ...
      results.train.acc, ...
      results.train.sens, ...
      results.train.spec, ...
    ] = getAccSensSpec(labels_train, top_train_predictions, true);
    [top_test_predictions, all_test_predictions] = getPredictionsFromModelOnImdb(net, 'cnn', imdb, 3);
    labels_test = imdb.images.labels(imdb.images.set == 3);
    [ ...
      results.test.acc, ...
      results.test.sens, ...
      results.test.spec, ...
    ] = getAccSensSpec(labels_test, top_test_predictions, true);
    saveStruct2File(results, opts.paths.results_file_path, 0);
  end
  results.info = info;

% -------------------------------------------------------------------------
function error_function = getErrorFunctionForDataset(dataset)
% -------------------------------------------------------------------------
  if strcmp(dataset, 'prostate-v2-20-patients') || ...
    strcmp(dataset, 'mnist-two-class-9-4') || ...
    strcmp(dataset, 'svhn-two-class-9-4') || ...
    strcmp(dataset, 'cifar-two-deer-horse') || ...
    strcmp(dataset, 'cifar-two-deer-truck')
    error_function = 'two-class';
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
