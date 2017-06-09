% -------------------------------------------------------------------------
function calculateDistances()
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
  %                                                                 Get IMDBs
  % -------------------------------------------------------------------------
  % dataset = 'cifar';
  % posneg_balance = 'whatever';
  dataset = 'cifar-multi-class-subsampled';
  posneg_balance = 'balanced-38';
  % dataset = 'cifar-two-class-deer-truck';
  % posneg_balance = 'balanced-38';

  fh_projection_utils = projectionUtils;
  experiments = {};

  afprintf(sprintf('[INFO] Loading original imdb...\n'));
  tmp_opts.dataset = dataset;
  tmp_opts.posneg_balance = posneg_balance;
  original_imdb = loadSavedImdb(tmp_opts, 1);
  % original_imdb = filterImdbForSet(original_imdb, 1, 1);
  afprintf(sprintf('[INFO] done!\n'));




  afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  projected_imdb = fh_projection_utils.getDenslyProjectedImdb(original_imdb);
  experiments{end+1}.imdb = projected_imdb;
  experiments{end}.title = 'Dense Random Projection Matrix';
  afprintf(sprintf('[INFO] done!\n'));


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV1P0RL0-ensemble-sparse-rp';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Random Gaussian V1P0 w/o ReLU';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV1P0RL1-ensemble-sparse-rp';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Random Gaussian V1P0 w ReLU';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV3P0RL0';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Random Gaussian V3P0 w/o ReLU';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV3P0RL3';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Random Gaussian V3P0 w/ ReLU';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV3P3RL0';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Random Gaussian V3P3 (LeNet) w/o ReLU';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV3P3RL3';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Random Gaussian V3P3 (LeNet) w/ ReLU';
  % afprintf(sprintf('[INFO] done!\n'));


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV1P0RL0', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained V1P0 w/o ReLU - trained on 38';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV1P0RL1', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 2);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained V1P0 w ReLU - trained on 38';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P0RL0', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 3);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained V3P0 w/o ReLU - trained on 38';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P0RL3', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 6);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained V3P0 w/ ReLU - trained on 38';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P3RL0', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 6);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained V3P3 (LeNet) w/o ReLU - trained on 38';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P3RL3', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 9);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained V3P3 (LeNet) w/ ReLU - trained on 38';
  % afprintf(sprintf('[INFO] done!\n'));


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % tmp = load(path_2);
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P3RL3', 'cifar', 'whatever');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 9);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained V3P3 (LeNet) - trained on ALL';
  % afprintf(sprintf('[INFO] done!\n'));

  for i = 1 : numel(experiments)
    input_opts = {};
    input_opts.dataset = dataset;
    input_opts.imdb = experiments{i}.imdb;
    [~, experiments{i}.performance_summary] = testKnn(input_opts);
  end

  for i = 1 : numel(experiments)
    afprintf(sprintf( ...
      '[INFO] 1-KNN Results for `%s`: \t\t train acc = %.4f, test acc = %.4f \n\n', ...
      experiments{i}.title, ...
      experiments{i}.performance_summary.testing.train.accuracy, ...
      experiments{i}.performance_summary.testing.test.accuracy));
  end

% -------------------------------------------------------------------------
function projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, projection_depth)
% -------------------------------------------------------------------------
  fh_projection_utils = projectionUtils;
  larp_weight_init_sequence = getLarpWeightInitSequence(larp_weight_init_type, larp_network_arch);
  projection_net = fh_projection_utils.getProjectionNetworkObject(dataset, larp_network_arch, larp_weight_init_sequence);
  projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, projection_depth);
