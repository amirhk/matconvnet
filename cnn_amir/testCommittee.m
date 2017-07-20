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
  opts.general.dataset = getValueFromFieldOrDefault(input_opts, 'dataset', 'mnist');
  opts.general.return_performance_summary = getValueFromFieldOrDefault(input_opts, 'return_performance_summary', true);
  opts.general.debug_flag = getValueFromFieldOrDefault(input_opts, 'debug_flag', true);

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
  opts.paths.options_file_path = fullfile(opts.paths.experiment_dir, '_options.txt');
  opts.paths.results_file_path = fullfile(opts.paths.experiment_dir, '_results.txt');

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
      opts.single_model_options.weight_init_sequence = getValueFromFieldOrDefault(input_opts, 'weight_init_sequence', {'gaussian', 'gaussian', 'gaussian'});
  end

  % -------------------------------------------------------------------------
  %                                                    save experiment setup!
  % -------------------------------------------------------------------------
  saveStruct2File(opts, opts.paths.options_file_path, 0);


  % -------------------------------------------------------------------------
  %                                                           train committee
  % -------------------------------------------------------------------------
  committee_models = {};
  opts.single_model_options.imdb = imdb;

  tic;
  for i = 1:opts.committee_options.number_of_committee_members
    afprintf(sprintf('[INFO] Training committee member #%d\n', i));
    switch opts.committee_options.training_method
      case 'ecocsvm'
        [model, ~] = testEcocSvm(opts.single_model_options);
      case 'cnn'
        [model, ~] = testCnn(opts.single_model_options);
    end
    committee_models{i} = model;
  end
  training_duration = toc;

  % -------------------------------------------------------------------------
  %                                                   get performance summary
  % -------------------------------------------------------------------------
  labels_train = imdb.images.labels(imdb.images.set == 1);
  labels_test = imdb.images.labels(imdb.images.set == 3);
  model_object = committee_models;
  model_string = sprintf('committee-%s', opts.committee_options.training_method);
  dataset = opts.general.dataset;

  % evaluate
  tic;
  [ ...
    train_accuracy, ...
    train_sensitivity, ...
    train_specificity, ...
  ] = getPerformanceSummary(model_object, model_string, dataset, imdb, labels_train, 'train', opts.general.return_performance_summary, opts.general.debug_flag);
  train_duration = toc;
  tic;
  [ ...
    test_accuracy, ...
    test_sensitivity, ...
    test_specificity, ...
  	] = getPerformanceSummary(model_object, model_string, dataset, imdb, labels_test, 'test', opts.general.return_performance_summary, opts.general.debug_flag);
  test_duration = toc;

  % -------------------------------------------------------------------------
  %                                                             assign output
  % -------------------------------------------------------------------------
  performance_summary.training.duration = training_duration;
  performance_summary.testing.train.accuracy = train_accuracy;
  performance_summary.testing.train.sensitivity = train_sensitivity;
  performance_summary.testing.train.specificity = train_specificity;
  performance_summary.testing.train.duration = train_duration;
  performance_summary.testing.test.accuracy = test_accuracy;
  performance_summary.testing.test.sensitivity = test_sensitivity;
  performance_summary.testing.test.specificity = test_specificity;
  performance_summary.testing.test.duration = test_duration;


  % -------------------------------------------------------------------------
  %                                                               save output
  % -------------------------------------------------------------------------
  saveStruct2File(performance_summary, opts.paths.results_file_path, 0);

