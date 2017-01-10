% -------------------------------------------------------------------------
function [trained_model, performance_summary] = testForest(input_opts)
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

  % -------------------------------------------------------------------------
  %                                                                 opts.imdb
  % -------------------------------------------------------------------------
  imdb = getValueFromFieldOrDefault(input_opts, 'imdb', struct());

  % -------------------------------------------------------------------------
  %                                                                opts.train
  % -------------------------------------------------------------------------
  opts.train.number_of_examples = size(imdb.images.data, 4);
  opts.train.number_of_features = 3072;
  opts.train.number_of_trees = getValueFromFieldOrDefault(input_opts, 'number_of_trees', 1000);
  opts.train.boosting_method = getValueFromFieldOrDefault(input_opts, 'boosting_method', 'RUSBoost'); % {'AdaBoostM1', 'RUSBoost'}

  % -------------------------------------------------------------------------
  %                                                                opts.paths
  % -------------------------------------------------------------------------
  opts.paths.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.paths.experiment_parent_dir = getValueFromFieldOrDefault( ...
    input_opts, ...
    'experiment_parent_dir', ...
    fullfile(vl_rootnn, 'experiment_results'));
  opts.paths.experiment_dir = fullfile(opts.paths.experiment_parent_dir, sprintf( ...
    'test-forest-%s-%s', ...
    opts.paths.time_string, ...
    opts.general.dataset, ...
    opts.train.boosting_method));
  if ~exist(opts.paths.experiment_dir)
    mkdir(opts.paths.experiment_dir);
  end
  opts.paths.options_file_path = fullfile(opts.paths.experiment_dir, 'options.txt');
  opts.paths.results_file_path = fullfile(opts.paths.experiment_dir, 'results.txt');

  % -------------------------------------------------------------------------
  %                                                    save experiment setup!
  % -------------------------------------------------------------------------
  saveStruct2File(opts, opts.paths.options_file_path, 0);

  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  vectorized_images = reshape(imdb.images.data, 3072, [])';
  labels = imdb.images.labels;
  Y = labels(1:opts.train.number_of_examples);
  is_train = imdb.images.set == 1;
  is_test = imdb.images.set == 3;

  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  printConsoleOutputSeparator();
  t = templateTree('MinLeafSize',5);
  tic
  rus_tree = fitensemble( ...
    vectorized_images(is_train,:), ...
    Y(is_train), ...
    opts.train.boosting_method, ...
    opts.train.number_of_trees, ...
    t, ...
    'LearnRate', 0.1, ...
    'nprint', 25);
  toc

  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  l_loss = loss(rus_tree, vectorized_images(is_test,:), Y(is_test), 'mode', 'cumulative');

  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  % figure;
  % tic
  % plot(l_loss);
  % toc
  % grid on;
  % xlabel('Number of trees');
  % ylabel('Test classification error');

  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  tic
  Yfit = predict(rus_tree, vectorized_images(is_test,:));
  toc
  % tab = tabulate(Y(is_test));
  % confusion_matrix = bsxfun(@rdivide, confusionmat(Y(is_test), Yfit), tab(:,2)) * 100;
  % acc = (1 - l_loss(end)) * 100;
  % spec = confusion_matrix(1,1);
  % sens = confusion_matrix(2,2);

  test_labels = Y(is_test);
  test_predictions = Yfit;
  [ ...
    acc, ...
    sens, ...
    spec, ...
  ] = getAccSensSpec(test_labels, test_predictions, true);
  printConsoleOutputSeparator();

  trained_model = rus_tree;
  performance_summary.weighted_test_accuracy = acc;
  performance_summary.weighted_test_sensitivity = sens;
  performance_summary.weighted_test_specificity = spec;
  saveStruct2File(performance_summary, opts.paths.results_file_path, 0);
