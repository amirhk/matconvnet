% -------------------------------------------------------------------------
function testElm()
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


  OUTPUT_SIZE = 100;
  % OUTPUT_SIZE = 16;
  % OUTPUT_SIZE = 4;

  dim_multiplier_list = [1,2,4,8];
  % dim_multiplier_list = [1,2];

  simple_train_accuracies = [];
  simple_test_accuracies = [];
  conv_train_accuracies = [];
  conv_test_accuracies = [];

  simple_test_1nns = [];
  conv_test_1nns = [];
  pca_test_1nns = ones(1, length(dim_multiplier_list)) * .7324;

  for i = 1 : length(dim_multiplier_list)
    dim_multiplier = dim_multiplier_list(i);
    [mean_simple_train_acc, mean_simple_test_acc, mean_simple_test_1nn, mean_conv_train_acc, mean_conv_test_acc, mean_conv_test_1nn] = mainBeef(dim_multiplier);

    simple_train_accuracies(end+1) = mean_simple_train_acc;
    simple_test_accuracies(end+1) = mean_simple_test_acc;
    conv_train_accuracies(end+1) = mean_conv_train_acc;
    conv_test_accuracies(end+1) = mean_conv_test_acc;

    simple_test_1nns(end+1) = mean_simple_test_1nn;
    conv_test_1nns(end+1) = mean_conv_test_1nn;
  end

  dimensionality = dim_multiplier_list * OUTPUT_SIZE;

  figure,

  subplot(1,2,1),
  plot( ...
    dimensionality, simple_train_accuracies, ...
    dimensionality, simple_test_accuracies, ...
    dimensionality, conv_train_accuracies, ...
    dimensionality, conv_test_accuracies, ...
    'LineWidth', 2);
  legend({'simple train accuracy', 'simple test accuracy', 'conv train accuracy', 'conv test accuracy'}, 'Location','northwest'),
  title('ELM Classification Performance'),

  subplot(1,2,2),
  plot( ...
    dimensionality, simple_test_1nns, ...
    dimensionality, conv_test_1nns, ...
    dimensionality, pca_test_1nns, ...
    'LineWidth', 2);
  legend({'simple test 1nns', 'conv test 1nns', 'pca test 1nns'}, 'Location','northwest'),
  title('1-NN Classification Performance on Final Layer ELM Embedding in 10-D'),


  dimensionality

  simple_train_accuracies
  simple_test_accuracies
  conv_train_accuracies
  conv_test_accuracies

  simple_test_1nns
  conv_test_1nns
  pca_test_1nns



% --------------------------------------------------------------------
function [mean_simple_train_acc, mean_simple_test_acc, mean_simple_test_1nn, mean_conv_train_acc, mean_conv_test_acc, mean_conv_test_1nn] = mainBeef(dim_multiplier)
% --------------------------------------------------------------------
  all_training_accuracy_simple = [];
  all_training_time_simple = [];
  all_testing_accuracy_simple = [];
  all_testing_time_simple = [];

  all_training_accuracy_conv = [];
  all_training_time_conv = [];
  all_testing_accuracy_conv = [];
  all_testing_time_conv = [];

  all_testing_1nn_simple = [];
  all_testing_1nn_conv = [];

  iterations = 10;
  % iterations = 30;

  for i = 1 : iterations
    afprintf(sprintf('\t Test #%d...', i));
    [training_time, testing_time, training_accuracy, testing_accuracy, final_layer_projected_imdb] = ELM(1, 'sig', 'simple', dim_multiplier);
    all_training_accuracy_simple(end+1) = training_accuracy;
    all_training_time_simple(end+1) = training_time;
    all_testing_accuracy_simple(end+1) = testing_accuracy;
    all_testing_time_simple(end+1) = testing_time;
    all_testing_1nn_simple(end+1) = get1NNperf(final_layer_projected_imdb);
    afprintf(sprintf('\t done.\n'));
  end

  for i = 1 : iterations
    afprintf(sprintf('\t Test #%d...', i));
    [training_time, testing_time, training_accuracy, testing_accuracy, final_layer_projected_imdb] = ELM(1, 'sig', 'conv', dim_multiplier);
    all_training_accuracy_conv(end+1) = training_accuracy;
    all_training_time_conv(end+1) = training_time;
    all_testing_accuracy_conv(end+1) = testing_accuracy;
    all_testing_time_conv(end+1) = testing_time;
    all_testing_1nn_conv(end+1) = get1NNperf(final_layer_projected_imdb);
    afprintf(sprintf('\t done.\n'));
  end

  % afprintf(sprintf('[SIMPLE] Train Accuracy Mean: %.3f, Std: %.3f\n', mean(all_training_accuracy_simple), std(all_training_accuracy_simple)));
  % afprintf(sprintf('[SIMPLE] Train Time Mean: %.3f, Std: %.3f\n', mean(all_training_time_simple), std(all_training_time_simple)));
  % afprintf(sprintf('[SIMPLE] Test Accuracy Mean: %.3f, Std: %.3f\n', mean(all_testing_accuracy_simple), std(all_testing_accuracy_simple)));
  % afprintf(sprintf('[SIMPLE] Test Time Mean: %.3f, Std: %.3f\n', mean(all_testing_time_simple), std(all_testing_time_simple)));

  % afprintf(sprintf('[CONV] Train Accuracy Mean: %.3f, Std: %.3f\n', mean(all_training_accuracy_conv), std(all_training_accuracy_conv)));
  % afprintf(sprintf('[CONV] Train Time Mean: %.3f, Std: %.3f\n', mean(all_training_time_conv), std(all_training_time_conv)));
  % afprintf(sprintf('[CONV] Test Accuracy Mean: %.3f, Std: %.3f\n', mean(all_testing_accuracy_conv), std(all_testing_accuracy_conv)));
  % afprintf(sprintf('[CONV] Test Time Mean: %.3f, Std: %.3f\n', mean(all_testing_time_conv), std(all_testing_time_conv)));

  mean_simple_train_acc = mean(all_training_accuracy_simple);
  mean_simple_test_acc = mean(all_testing_accuracy_simple);
  mean_simple_test_1nn = mean(all_testing_1nn_simple);
  mean_conv_train_acc = mean(all_training_accuracy_conv);
  mean_conv_test_acc = mean(all_testing_accuracy_conv);
  mean_conv_test_1nn = mean(all_testing_1nn_conv);


% --------------------------------------------------------------------
function performance = get1NNperf(imdb)
% --------------------------------------------------------------------
  experiment_options.imdb = imdb;
  experiment_options.dataset = 'mnist-fashion-multi-class-subsampled';
  experiment_options.posneg_balance = 'balanced-250';
  performance = getSimpleTestAccuracyFromKnn(experiment_options);
