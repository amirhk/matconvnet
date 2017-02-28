% -------------------------------------------------------------------------
function runEnsembleLarpTests(dataset, posneg_balance, projection, gpus)
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
  opts.general.dataset = dataset;
  opts.general.network_arch = 'lenet';

  % -------------------------------------------------------------------------
  %                                                                 opts.imdb
  % -------------------------------------------------------------------------
  opts.imdb.posneg_balance = posneg_balance;
  opts.imdb.projection = projection;

  % -------------------------------------------------------------------------
  %                                                                opts.train
  % -------------------------------------------------------------------------
  opts.train.gpus = gpus;

  % -------------------------------------------------------------------------
  %                                                                opts.paths
  % -------------------------------------------------------------------------
  opts.paths.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.paths.experiment_parent_dir = getValueFromFieldOrDefault( ...
    {}, ... % TODO: this should be input_opts
    'experiment_parent_dir', ...
    fullfile(vl_rootnn, 'experiment_results'));
  opts.paths.experiment_dir = fullfile(opts.paths.experiment_parent_dir, sprintf( ...
    'test-ensemble-larp-tests-%s-%s-%s-GPU-%d', ...
    opts.paths.time_string, ...
    opts.general.dataset, ...
    opts.imdb.posneg_balance, ...
    opts.train.gpus));
  if ~exist(opts.paths.experiment_dir)
    mkdir(opts.paths.experiment_dir);
  end
  opts.paths.options_file_path = fullfile(opts.paths.experiment_dir, 'options.txt');
  % opts.paths.results_file_path = fullfile(opts.paths.experiment_dir, 'results.txt');

  % -------------------------------------------------------------------------
  %                                                    save experiment setup!
  % -------------------------------------------------------------------------
  saveStruct2File(opts, opts.paths.options_file_path, 0);

  % TODO:
  % ~~~    1. experiment_parent_dir code
  % ###    2. merge test*.m files (with shared loop function)

  % -------------------------------------------------------------------------
  %                                                            shared options
  % -------------------------------------------------------------------------
  experiment_options.number_of_folds = 3;
  experiment_options.experiment_parent_dir = opts.paths.experiment_dir;
  experiment_options.dataset = opts.general.dataset;
  experiment_options.network_arch = opts.general.network_arch;
  experiment_options.posneg_balance = opts.imdb.posneg_balance;
  experiment_options.projection = opts.imdb.projection;
  experiment_options.gpus = opts.train.gpus;


  % % -------------------------------------------------------------------------
  % %                                                                single svm
  % % -------------------------------------------------------------------------
  % experiment_options.training_method = 'svm';
  % % Exp. 1
  % testKFold(experiment_options);

  % % -------------------------------------------------------------------------
  % %                                                             single libsvm
  % % -------------------------------------------------------------------------
  % experiment_options.training_method = 'libsvm';
  % % Exp. i
  % % for c = logspace(-2,3,6)
  % for i = -3:1:5
  %   c = 2^i;
  %   experiment_options.libsvm_options = sprintf('-q -t 0 -c %f', c);
  %   testKFold(experiment_options);
  % end

  % % -------------------------------------------------------------------------
  % %                                                             single libsvm
  % % -------------------------------------------------------------------------
  experiment_options.training_method = 'minfuncsvm';
  % Exp. i
  % for max_iters = [100, 1000]
  for max_iters = [1000]
    % for c = logspace(-7,3,11)
    for c = logspace(-5,-4,5)
    % for c = logspace(-7,-1,7)
    % for c = logspace(-2,3,6)
    % for i = -3:1:5
    %   c = 2^i;
      experiment_options.minfuncsvm_c_penalty = c;
      experiment_options.minfuncsvm_max_iters = max_iters;
      testKFold(experiment_options);
    end
  end

  % % -------------------------------------------------------------------------
  % %                                                              ensemble svm
  % % -------------------------------------------------------------------------
  % experiment_options.training_method = 'ensemble-svm';
  % experiment_options.boosting_method = 'adaboost.m1';
  % experiment_options.iteration_count = 8;

  % % Exp. 1
  % experiment_options.loss_calculation_method = 'default_in_literature';
  % testKFold(experiment_options);

  % % Exp. 2
  % experiment_options.loss_calculation_method = 'class_normalized';
  % testKFold(experiment_options);

  % % -------------------------------------------------------------------------
  % %                                                               single tree
  % % -------------------------------------------------------------------------
  % TODO...

  % % -------------------------------------------------------------------------
  % %                                                                    forest
  % % -------------------------------------------------------------------------
  % experiment_options.training_method = 'forest';
  % % % Exp. 1
  % % experiment_options.boosting_method = 'AdaBoostM1';
  % % testKFold(experiment_options);
  % % Exp. 2
  % experiment_options.boosting_method = 'RUSBoost';
  % testKFold(experiment_options);

  % % -------------------------------------------------------------------------
  % %                                                                single cnn
  % % -------------------------------------------------------------------------
  % experiment_options.training_method = 'single-mlp';
  % testKFold(experiment_options);

  % % -------------------------------------------------------------------------
  % %                                                                single cnn
  % % -------------------------------------------------------------------------
  experiment_options.training_method = 'single-cnn';

  % % -------------------------------------------------
  % experiment_options.network_arch = 'lenet';
  % % -------------------------------------------------

  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % Exp. 2
  % experiment_options.backprop_depth = 7;
  % testKFold(experiment_options);

  % % Exp. 3
  % experiment_options.backprop_depth = 10;
  % testKFold(experiment_options);

  % % Exp. 4
  % experiment_options.backprop_depth = 13;
  % testKFold(experiment_options);


  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV0P0+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV1P0+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV1P1+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV3P0+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV3P1+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV3P3+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV5hP0+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV5hP1+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV5hP3+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV5hP5+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV5aP0+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV5aP1+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV5aP3+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV5aP5+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);


%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%


  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV0P0+convV0P0+fcV2';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 6;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV1P0+convV0P0+fcV2';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 6;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV1P1+convV0P0+fcV2';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 6;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV3P0+convV0P0+fcV2';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 6;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV3P1+convV0P0+fcV2';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 6;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV3P3+convV0P0+fcV2';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 6;
  % testKFold(experiment_options);


%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%


  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV0sP0+convV1sP1+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 7;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV1sP0+convV1sP1+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 7;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV2sP0+convV1sP1+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 7;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV1lP0+convV1lP1+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 7;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV2lP0+convV1lP1+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 7;
  % testKFold(experiment_options);


%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%


  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV0P0+convV3lP1+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 11;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV1lP0+convV3lP1+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 11;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV2lP0+convV3lP1+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 11;
  % testKFold(experiment_options);


%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV0P0+convV3P3+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 13;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV1lP0+convV3P3+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 13;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV1lP1+convV3P3+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 13;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV2lP0+convV3P3+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 13;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV2lP1+convV3P3+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 13;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV2lP2+convV3P3+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 13;
  % testKFold(experiment_options);





  % % -------------------------------------------------
  % experiment_options.network_arch = 'TMP_NETWORK';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 13;
  % testKFold(experiment_options);










  % % -------------------------------------------------
  % experiment_options.network_arch = 'TODO fc_lenet_with_larger_fc_conv';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 6;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'lenet_with_larger_fc_conv';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 6;
  % testKFold(experiment_options);




  % % -------------------------------------------------------------------------
  % %                                                             committee cnn
  % % -------------------------------------------------------------------------
  % experiment_options.training_method = 'committee-cnn';

  % % -------------------------------------------------
  % experiment_options.number_of_committee_members = 3;
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % Exp. 2
  % experiment_options.backprop_depth = 7;
  % testKFold(experiment_options);

  % % Exp. 3
  % experiment_options.backprop_depth = 13;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.number_of_committee_members = 7;
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % Exp. 2
  % experiment_options.backprop_depth = 7;
  % testKFold(experiment_options);

  % % Exp. 3
  % experiment_options.backprop_depth = 13;
  % testKFold(experiment_options);


  % % -------------------------------------------------------------------------
  % %                                                              ensemble cnn
  % % -------------------------------------------------------------------------
  % experiment_options.training_method = 'ensemble-cnn';
  % experiment_options.iteration_count = 8;
  % experiment_options.number_of_samples_per_model = 1000;
  % experiment_options.uni_model_boosting = false;

  % % -------------------------------------------------
  % experiment_options.boosting_method = 'rusboost';
  % % -------------------------------------------------

  % % Exp. 1
  % experiment_options.loss_calculation_method = 'default_in_literature';
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % Exp. 2
  % experiment_options.loss_calculation_method = 'default_in_literature';
  % experiment_options.backprop_depth = 7;
  % testKFold(experiment_options);

  % % Exp. 3
  % experiment_options.loss_calculation_method = 'default_in_literature';
  % experiment_options.backprop_depth = 13;
  % testKFold(experiment_options);

  % % Exp. 4
  % experiment_options.loss_calculation_method = 'class_normalized';
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % Exp. 5
  % experiment_options.loss_calculation_method = 'class_normalized';
  % experiment_options.backprop_depth = 7;
  % testKFold(experiment_options);

  % % Exp. 6
  % experiment_options.loss_calculation_method = 'class_normalized';
  % experiment_options.backprop_depth = 13;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.boosting_method = 'adaboost.m1';
  % % -------------------------------------------------

  % % Exp. 1
  % experiment_options.loss_calculation_method = 'default_in_literature';
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % % Exp. 2
  % % experiment_options.loss_calculation_method = 'default_in_literature';
  % % experiment_options.backprop_depth = 7;
  % % testKFold(experiment_options);

  % % Exp. 3
  % experiment_options.loss_calculation_method = 'default_in_literature';
  % experiment_options.backprop_depth = 13;
  % testKFold(experiment_options);

  % % Exp. 4
  % experiment_options.loss_calculation_method = 'class_normalized';
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % Exp. 5
  % experiment_options.loss_calculation_method = 'class_normalized';
  % experiment_options.backprop_depth = 7;
  % testKFold(experiment_options);

  % % Exp. 6
  % experiment_options.loss_calculation_method = 'class_normalized';
  % experiment_options.backprop_depth = 13;
  % testKFold(experiment_options);


  % % -------------------------------------------------
  % experiment_options.boosting_method = 'adaboost.m1';
  % % -------------------------------------------------
  % experiment_options.network_arch = 'alexnet';

  % % Exp. 1
  % experiment_options.loss_calculation_method = 'default_in_literature';
  % experiment_options.backprop_depth = 7;
  % testKFold(experiment_options);

  % % Exp. 3
  % experiment_options.loss_calculation_method = 'default_in_literature';
  % experiment_options.backprop_depth = 20;
  % testKFold(experiment_options);



  % % -------------------------------------------------------------------------
  % %                                                      ensemble multi-class
  % % -------------------------------------------------------------------------
  % experiment_options.training_method = 'ensemble-multi-class-cnn';
  % % experiment_options.training_method = 'ensemble-multi-class-svm';
  % experiment_options.iteration_count = 8;
  % experiment_options.uni_model_boosting = false;

  % experiment_options.boosting_method = 'adaboost.m1';

  % % -------------------------------------------------
  % experiment_options.number_of_samples_per_model = 100000;
  % % -------------------------------------------------

  % % Exp. 1
  % experiment_options.loss_calculation_method = 'default_in_literature';
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % Exp. 1
  % experiment_options.loss_calculation_method = 'default_in_literature';
  % experiment_options.backprop_depth = 13;
  % testKFold(experiment_options);

















