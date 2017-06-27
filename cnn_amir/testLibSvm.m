% -------------------------------------------------------------------------
function [trained_model, performance_summary] = testLibSvm(input_opts)
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
  %                                                                opts.train
  % -------------------------------------------------------------------------
  opts.train.number_of_features = prod(size(imdb.images.data(:,:,:,1))); % 32 x 32 x 3 = 3072
  opts.train.libsvm_options = getValueFromFieldOrDefault(input_opts, 'libsvm_options', '-t 0');

  % -------------------------------------------------------------------------
  %                                                                opts.paths
  % -------------------------------------------------------------------------
  opts.paths.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.paths.experiment_parent_dir = getValueFromFieldOrDefault( ...
    input_opts, ...
    'experiment_parent_dir', ...
    fullfile(vl_rootnn, 'experiment_results'));
  opts.paths.experiment_dir = fullfile(opts.paths.experiment_parent_dir, sprintf( ...
    'libsvm-%s-%s-libsvm-options%s', ...
    opts.paths.time_string, ...
    opts.general.dataset, ...
    strrep(opts.train.libsvm_options,' ','_'))); % '-t 0 -c 10' --> '-t_0_-c_10'
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
  labels_train = labels(is_train);
  labels_test = labels(is_test);

  % -------------------------------------------------------------------------
  %                                                                     train
  % -------------------------------------------------------------------------
  % TODO: this format below is specific for libsvm / minfuncsvm... can maybe
  % unify with svm and forest later...
  vectorized_data_train = double(vectorized_data_train);
  labels_train = double(labels_train);
  labels_train = labels_train';

  vectorized_data_test = double(vectorized_data_test);
  labels_test = double(labels_test);
  labels_test = labels_test';

  tic;
  libsvm_struct = svmtrain(labels_train, vectorized_data_train, opts.train.libsvm_options);
  training_duration = toc;

  if isTwoClassImdb(opts.general.dataset)
    fhGetAccSensSpec = @getAccSensSpec;
  else
    fhGetAccSensSpec = @getAccSensSpecMultiClass;
  end

  % -------------------------------------------------------------------------
  %                                                   get performance summary
  % -------------------------------------------------------------------------
  model_object = libsvm_struct;
  model_string = 'libsvm';
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
  trained_model = libsvm_struct;
  performance_summary.training.duration = training_duration;
  performance_summary.testing.train.accuracy = train_accuracy;
  performance_summary.testing.train.sensitivity = train_sensitivity;
  performance_summary.testing.train.specificity = train_specificity;
  performance_summary.testing.train.duration = train_duration;
  performance_summary.testing.test.accuracy = test_accuracy;
  performance_summary.testing.test.sensitivity = test_sensitivity;
  performance_summary.testing.test.specificity = test_specificity;
  performance_summary.testing.test.duration = test_duration;
  % TODO: add more fields here... time it took to run experiment, as well as optional field to pass back anything to testKFold which saves stuff!!!!!!

  % -------------------------------------------------------------------------
  %                                                               save output
  % -------------------------------------------------------------------------
  saveStruct2File(performance_summary, opts.paths.results_file_path, 0);
