% -------------------------------------------------------------------------
function [trained_model, performance_summary] = testCommittee(input_opts)
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

  % -------------------------------------------------------------------------
  %                                                              opts.general
  % -------------------------------------------------------------------------
  opts.general.dataset = getValueFromFieldOrDefault(input_opts, 'dataset', 'mnist-two-class-9-4');
  opts.general.return_performance_summary = getValueFromFieldOrDefault(input_opts, 'return_performance_summary', true);

  % -------------------------------------------------------------------------
  %                                                                 opts.imdb
  % -------------------------------------------------------------------------
  imdb = getValueFromFieldOrDefault(input_opts, 'imdb', struct());

  % -------------------------------------------------------------------------
  %                                                    opts.committee_options
  % -------------------------------------------------------------------------
  opts.committee_options.number_of_committee_members = getValueFromFieldOrDefault(input_opts, 'number_of_committee_members', 3);
  opts.committee_options.training_method = getValueFromFieldOrDefault(input_opts, 'training_method', 'cnn');

  % -------------------------------------------------------------------------
  %                                                                opts.paths
  % -------------------------------------------------------------------------
  opts.paths.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.paths.experiment_parent_dir = getValueFromFieldOrDefault( ...
    input_opts, ...
    'experiment_parent_dir', ...
    fullfile(vl_rootnn, 'experiment_results'));
  opts.paths.experiment_dir = fullfile(opts.paths.experiment_parent_dir, sprintf( ...
    'committee-%d-%s-%s-%s', ...
    opts.committee_options.number_of_committee_members, ...
    opts.committee_options.training_method, ...
    opts.paths.time_string, ...
    opts.general.dataset));
  if ~exist(opts.paths.experiment_dir)
    mkdir(opts.paths.experiment_dir);
  end
  opts.paths.options_file_path = fullfile(opts.paths.experiment_dir, 'options.txt');
  opts.paths.results_file_path = fullfile(opts.paths.experiment_dir, 'results.txt');

  % -------------------------------------------------------------------------
  %                                                 opts.single_model_options
  % -------------------------------------------------------------------------
  opts.single_model_options.dataset = opts.general.dataset;
  opts.single_model_options.experiment_parent_dir = opts.paths.experiment_dir;
  opts.single_model_options.return_performance_summary = true;
  switch opts.committee_options.training_method
    case 'svm'
      % no additional options
    case 'cnn'
      opts.single_model_options.network_arch = getValueFromFieldOrDefault(input_opts, 'network_arch', 'lenet');
      opts.single_model_options.backprop_depth = getValueFromFieldOrDefault(input_opts, 'backprop_depth', 4);
      opts.single_model_options.gpus = ifNotMacSetGpu(getValueFromFieldOrDefault(input_opts, 'gpus', 1));
      opts.single_model_options.debug_flag = getValueFromFieldOrDefault(input_opts, 'debug_flag', false);
      opts.single_model_options.learning_rate = getValueFromFieldOrDefault(input_opts, 'learning_rate', 'default_keyword');
      opts.single_model_options.weight_init_sequence = getValueFromFieldOrDefault(input_opts, 'weight_init_sequence', {'compRand', 'compRand', 'compRand'});
  end

  % -------------------------------------------------------------------------
  %                                                    save experiment setup!
  % -------------------------------------------------------------------------
  saveStruct2File(opts, opts.paths.options_file_path, 0);


  % -------------------------------------------------------------------------
  %                                                        5. Train committee
  % -------------------------------------------------------------------------
  committee_models = {};
  opts.single_model_options.imdb = imdb;

  for i = 1:opts.committee_options.number_of_committee_members
    afprintf(sprintf('[INFO] Training committee member #%d\n', i));
    switch opts.committee_options.training_method
      case 'svm'
        [model, ~] = testSvm(opts.single_model_options);
      case 'cnn'
        [model, ~] = testCnn(opts.single_model_options);
    end
    committee_models{i} = model;
  end

  % -------------------------------------------------------------------------
  %                                                   get performance summary
  % -------------------------------------------------------------------------
  training_method = sprintf('committee-%s', opts.committee_options.training_method);
  if opts.general.return_performance_summary
    afprintf(sprintf('[INFO] Getting model performance on `train` set...\n'));
    [top_train_predictions, ~] = getPredictionsFromModelOnImdb(committee_models, training_method, imdb, 1);
    afprintf(sprintf('[INFO] Model performance on `train` set\n'));
    labels_train = imdb.images.labels(imdb.images.set == 1);
    [ ...
      train_accuracy, ...
      train_sensitivity, ...
      train_specificity, ...
    ] = getAccSensSpec(labels_train, top_train_predictions, true);
    afprintf(sprintf('[INFO] Getting model performance on `test` set...\n'));
    [top_test_predictions, ~] = getPredictionsFromModelOnImdb(committee_models, training_method, imdb, 3);
    afprintf(sprintf('[INFO] Model performance on `test` set\n'));
    labels_test = imdb.images.labels(imdb.images.set == 3);
    [ ...
      test_accuracy, ...
      test_sensitivity, ...
      test_specificity, ...
    ] = getAccSensSpec(labels_test, top_test_predictions, true);
    printConsoleOutputSeparator();
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
  trained_model = committee_models;
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

