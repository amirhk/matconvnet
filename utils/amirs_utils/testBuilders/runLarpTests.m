% -------------------------------------------------------------------------
% function runLarpTests(experiment_parent_dir, dataset, posneg_balance, projection_arch, projection_kernel_generation, gpus)
function runLarpTests(experiment_parent_dir, dataset, posneg_balance, network_arch, larp_weight_init_type, projection, gpus)
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

  % -------------------------------------------------------------------------
  %                                                                 opts.imdb
  % -------------------------------------------------------------------------
  opts.imdb.posneg_balance = posneg_balance;
  opts.imdb.projection = projection;

  % -------------------------------------------------------------------------
  %                                                         opts.network_arch
  % -------------------------------------------------------------------------
  opts.net.network_arch = network_arch;
  opts.net.larp_weight_init_type = larp_weight_init_type;

  % -------------------------------------------------------------------------
  %                                                                opts.train
  % -------------------------------------------------------------------------
  opts.train.gpus = gpus;

  % -------------------------------------------------------------------------
  %                                                                opts.paths
  % -------------------------------------------------------------------------
  opts.paths.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.paths.experiment_parent_dir = experiment_parent_dir;
  opts.paths.experiment_dir = fullfile(opts.paths.experiment_parent_dir, sprintf( ...
    'test-larp-tests-%s-%s-%s-GPU-%d', ...
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
  experiment_options.number_of_folds = 1;
  experiment_options.experiment_parent_dir = opts.paths.experiment_dir;
  experiment_options.dataset = opts.general.dataset;
  experiment_options.posneg_balance = opts.imdb.posneg_balance;
  experiment_options.projection = opts.imdb.projection;
  experiment_options.gpus = opts.train.gpus;


  % % -------------------------------------------------------------------------
  % %                                                            single ecocsvm
  % % -------------------------------------------------------------------------
  % experiment_options.training_method = 'svm';
  % % Exp. 1
  % testKFold(experiment_options);

  % -------------------------------------------------------------------------
  %                                                             single libsvm
  % -------------------------------------------------------------------------
  % experiment_options.training_method = 'libsvm';
  % % Exp. i
  % % for c = logspace(-2,3,6)
  % % for i = -3:1:5
  % for c = logspace(-6,3,10)
  %   % c = 2^i;
  %   experiment_options.libsvm_options = sprintf('-q -t 0 -c %f', c);
  %   testKFold(experiment_options);
  % end

  % % -------------------------------------------------------------------------
  % %                                                         single minfuncsvm
  % % -------------------------------------------------------------------------
  % experiment_options.training_method = 'minfuncsvm';
  % % Exp. i
  % % for max_iters = [100, 1000]
  % for max_iters = [2500]
  %   for c = logspace(-6,3,10)
  %   % for c = logspace(-5,-4,5)
  %   % for c = logspace(-7,-1,7)
  %   % for c = logspace(-2,3,6)
  %   % for i = -3:1:5
  %   %   c = 2^i;
  %     experiment_options.minfuncsvm_c_penalty = c;
  %     experiment_options.minfuncsvm_max_iters = max_iters;
  %     testKFold(experiment_options);
  %   end
  % end

  % % -------------------------------------------------------------------------
  % %                                                          ensemble ecocsvm
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


  % if strcmp(projection, 'no-projection')
  %   something = 'larpV0P0SF'; % or 'larpV0P0ST'
  % else
  %   something = projection(19:end); %'projected-through-XXX' --> 'XXX'
  % end
  % conv_arch = getMatchingConvArchitectureForLarpArchitecture(something, 'v2');
  % experiment_options.network_arch = conv_arch;
  % experiment_options.backprop_depth = getFullBackPropDepthForConvArchitecture(conv_arch);

  % experiment_options.batch_size = 100;

  % base_learning_rate = [0.1*ones(1,25) 0.03*ones(1,25) 0.01*ones(1,50)];
  % % base_learning_rate = [0.1*ones(1,5)];
  % for learning_rate_divider = [1, 3, 10, 30]
  %   experiment_options.learning_rate = base_learning_rate / learning_rate_divider;
  %   for weight_decay = [0.01, 0.001, 0.0001]
  %     experiment_options.weight_decay = weight_decay;
  %     testKFold(experiment_options);
  %   end
  % end


% if strcmp(projection, 'no-projection')
  %   something = 'larpV0P0SF'; % or 'larpV0P0ST'
  % else
  %   something = projection(19:end); %'projected-through-XXX'
  % end
  % experiment_options.network_arch = strcat(something, '+convV0P0+fcV1');
  % % conv_arch = getMatchingConvArchitectureForLarpArchitecture(something, 'v1');
  % % experiment_options.network_arch = conv_arch;
  % % experiment_options.backprop_depth = getFullBackPropDepthForConvArchitecture(conv_arch);
  % % network_arch = projection;

  % experiment_options.batch_size = 100;

  % base_learning_rate = [0.1*ones(1,25) 0.03*ones(1,25) 0.01*ones(1,50)];
  % % base_learning_rate = [0.1*ones(1,5)];
  % for learning_rate_divider = [3] % [1, 3, 10, 30]
  %   experiment_options.learning_rate = base_learning_rate / learning_rate_divider;
  %   for weight_decay = [0.0001] % [0.01, 0.001, 0.0001]
  %     experiment_options.weight_decay = weight_decay;
  %     testKFold(experiment_options);
  %   end
  % end

  experiment_options.network_arch = network_arch;
  experiment_options.backprop_depth = getFullBackPropDepthForNetworkArch(network_arch);
  experiment_options.weight_init_sequence = getWeightInitSequenceForWeightInitTypeAndNetworkArch(larp_weight_init_type, network_arch);

  % experiment_options.learning_rate = [0.1*ones(1,25) 0.03*ones(1,25) 0.01*ones(1,50)] / 10;
  % experiment_options.weight_decay = 0.0001;
  experiment_options.batch_size = 100;

  base_learning_rate = [0.1*ones(1,25) 0.03*ones(1,25) 0.01*ones(1,50)];
  % base_learning_rate = [0.1*ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)];
  % base_learning_rate = [0.1*ones(1,5)];
  % for learning_rate_divider = [10] % [1, 3, 10, 30]
  for learning_rate_divider = [1, 3, 10, 30]
    experiment_options.learning_rate = base_learning_rate / learning_rate_divider;
    % for weight_decay = [0.0001] % [0.01, 0.001, 0.0001]
    for weight_decay = [0.01, 0.001, 0.0001]
      experiment_options.weight_decay = weight_decay;
      testKFold(experiment_options);
    end
  end













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
  % experiment_options.network_arch = 'larpV0P0SF+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV1P0SF+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV1P1SF+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV3P0SF+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV3P1SF+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV3P3SF+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV5hP0SF+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV5hP1SF+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV5hP3SF+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV5hP5SF+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV5aP0SF+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV5aP1SF+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV5aP3SF+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV5aP5SF+convV0P0+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);


%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%


  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV0P0SF+convV0P0+fcV2';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 6;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV1P0SF+convV0P0+fcV2';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 6;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV1P1SF+convV0P0+fcV2';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 6;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV3P0SF+convV0P0+fcV2';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 6;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV3P1SF+convV0P0+fcV2';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 6;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV3P3SF+convV0P0+fcV2';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 6;
  % testKFold(experiment_options);


%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%


  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV0sP0SF+convV1sP1+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 7;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV1sP0SF+convV1sP1+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 7;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV2sP0SF+convV1sP1+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 7;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV1lP0SF+convV1lP1+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 7;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV2lP0SF+convV1lP1+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 7;
  % testKFold(experiment_options);


%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%


  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV0P0SF+convV3lP1+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 11;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV1lP0SF+convV3lP1+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 11;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV2lP0SF+convV3lP1+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 11;
  % testKFold(experiment_options);


%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV0P0SF+convV3P3+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 13;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV1lP0SF+convV3P3+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 13;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV1lP1SF+convV3P3+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 13;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV2lP0SF+convV3P3+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 13;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV2lP1SF+convV3P3+fcV1';
  % % -------------------------------------------------
  % % Exp. 1
  % experiment_options.backprop_depth = 13;
  % testKFold(experiment_options);

  % % -------------------------------------------------
  % experiment_options.network_arch = 'larpV2lP2SF+convV3P3+fcV1';
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

















