% -------------------------------------------------------------------------
function [trained_model, performance_summary] = testMlp(input_opts)
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

  % -------------------------------------------------------------------------
  %                                                                 opts.imdb
  % -------------------------------------------------------------------------
  imdb = getValueFromFieldOrDefault(input_opts, 'imdb', struct());

  % -------------------------------------------------------------------------
  %                                                                opts.train
  % -------------------------------------------------------------------------
  opts.train.number_of_features = prod(size(imdb.images.data(:,:,:,1))); % 32 x 32 x 3 = 3072
  opts.train.number_of_hidden_nodes = getValueFromFieldOrDefault(input_opts, 'number_of_hidden_nodes', [64 10]);

  % -------------------------------------------------------------------------
  %                                                                opts.paths
  % -------------------------------------------------------------------------
  opts.paths.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.paths.experiment_parent_dir = getValueFromFieldOrDefault( ...
    input_opts, ...
    'experiment_parent_dir', ...
    fullfile(vl_rootnn, 'experiment_results'));
  opts.paths.experiment_dir = fullfile(opts.paths.experiment_parent_dir, sprintf( ...
    'mlp-%s-%s', ...
    opts.paths.time_string, ...
    opts.general.dataset));
  if ~exist(opts.paths.experiment_dir)
    mkdir(opts.paths.experiment_dir);
  end
  opts.paths.options_file_path = fullfile(opts.paths.experiment_dir, 'options.txt');
  opts.paths.results_file_path = fullfile(opts.paths.experiment_dir, 'results.txt');

  % -------------------------------------------------------------------------
  %                                                    save experiment setup!
  % -------------------------------------------------------------------------
  saveStruct2File(opts, opts.paths.options_file_path, 0);

  % -------------------------------------------------------------------------
  %                                                   prepare data and labels
  % -------------------------------------------------------------------------
  vectorized_data = getVectorizedDataFromImdb(imdb);
  labels = imdb.images.labels;
  is_train = imdb.images.set == 1;
  is_test = imdb.images.set == 3;

  vectorized_data_train = vectorized_data(is_train, :);
  vectorized_data_test = vectorized_data(is_test, :);
  vectorized_data_train = vectorized_data_train';
  vectorized_data_test = vectorized_data_test';
  labels_train = labels(is_train);
  labels_test = labels(is_test);

  labels_train = reshape(labels_train, 1, []);
  labels_test = reshape(labels_test, 1, []);

  % -------------------------------------------------------------------------
  %                                                                     train
  % -------------------------------------------------------------------------

  % net = patternnet([64,10]);
  % net = patternnet([500,100]);
  net = patternnet(opts.train.number_of_hidden_nodes);
  tic;
  assert(size(vectorized_data_train, 1) == opts.train.number_of_features);
  assert(size(vectorized_data_test, 1) == opts.train.number_of_features);
  labels_train_matrix_form = full(ind2vec(double(labels_train)));

  if ispc
    net = train(net, vectorized_data_train, labels_train_matrix_form, 'useGPU', 'no', 'showResources', 'yes');
  else
    net = train(net, vectorized_data_train, labels_train_matrix_form, 'useGPU', 'no', 'showResources', 'yes');
  end
  training_duration = toc;

  % -------------------------------------------------------------------------
  %                                                   get performance summary
  % -------------------------------------------------------------------------

  model_object = net;
  model_string = 'mlp';
  dataset = opts.general.dataset;

  % evaluate
  tic;
  [ ...
    train_accuracy, ...
    train_sensitivity, ...
    train_specificity, ...
  ] = getPerformanceSummary(model_object, model_string, dataset, imdb, labels_train, 'train', true);
  train_duration = toc;
  tic;
  [ ...
    test_accuracy, ...
    test_sensitivity, ...
    test_specificity, ...
  ] = getPerformanceSummary(model_object, model_string, dataset, imdb, labels_test, 'test', true);
  test_duration = toc;

  % -------------------------------------------------------------------------
  %                                                             assign output
  % -------------------------------------------------------------------------
  trained_model = net;
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
