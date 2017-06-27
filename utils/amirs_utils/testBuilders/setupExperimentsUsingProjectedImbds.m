% -------------------------------------------------------------------------
function [original_imdb, experiments] = setupExperimentsUsingProjectedImbds(dataset, posneg_balance, should_filter_out_test_set)
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
  fh_projection_utils = projectionUtils;
  experiments = {};

  afprintf(sprintf('[INFO] Loading original imdb...\n'));
  tmp_opts.dataset = dataset;
  tmp_opts.posneg_balance = posneg_balance;
  original_imdb = loadSavedImdb(tmp_opts, 1);
  if should_filter_out_test_set
    original_imdb = filterImdbForSet(original_imdb, 1, 1);
  end
  afprintf(sprintf('[INFO] done!\n'));


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  experiments{end+1}.imdb = original_imdb;
  experiments{end}.title = 'Original IMDB';


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projected_imdb = fh_projection_utils.getDenslyProjectedImdb(original_imdb, 1, 0);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Dense RP = 1 - ReLU = 0';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projected_imdb = fh_projection_utils.getDenslyProjectedImdb(original_imdb, 2, 0);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Dense RP = 2 - ReLU = 0';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projected_imdb = fh_projection_utils.getDenslyProjectedImdb(original_imdb, 3, 0);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Dense RP = 3 - ReLU = 0';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projected_imdb = fh_projection_utils.getDenslyProjectedImdb(original_imdb, 4, 0);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Dense RP = 4 - ReLU = 0';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projected_imdb = fh_projection_utils.getDenslyProjectedImdb(original_imdb, 5, 0);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Dense RP = 5 - ReLU = 0';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projected_imdb = fh_projection_utils.getDenslyProjectedImdb(original_imdb, 1, 1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Dense RP = 1 - ReLU = 1';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projected_imdb = fh_projection_utils.getDenslyProjectedImdb(original_imdb, 2, 2);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Dense RP = 2 - ReLU = 2';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projected_imdb = fh_projection_utils.getDenslyProjectedImdb(original_imdb, 3, 3);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Dense RP = 3 - ReLU = 3';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projected_imdb = fh_projection_utils.getDenslyProjectedImdb(original_imdb, 4, 4);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Dense RP = 4 - ReLU = 4';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projected_imdb = fh_projection_utils.getDenslyProjectedImdb(original_imdb, 5, 5);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Dense RP = 5 - ReLU = 5';
  % afprintf(sprintf('[INFO] done!\n'));


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  angle_separated_imdb = fh_projection_utils.getAngleSeparatedImdb(original_imdb);
  experiments{end+1}.imdb = angle_separated_imdb;
  experiments{end}.title = 'Angle Separated Imdb';
  afprintf(sprintf('[INFO] done!\n'));


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  projected_imdb = fh_projection_utils.getDenslyProjectedImdb(angle_separated_imdb, 1, 0);
  experiments{end+1}.imdb = projected_imdb;
  experiments{end}.title = 'Dense RP = 1 - ReLU = 0';
  afprintf(sprintf('[INFO] done!\n'));

  afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  projected_imdb = fh_projection_utils.getDenslyProjectedImdb(angle_separated_imdb, 2, 0);
  experiments{end+1}.imdb = projected_imdb;
  experiments{end}.title = 'Dense RP = 2 - ReLU = 0';
  afprintf(sprintf('[INFO] done!\n'));

  afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  projected_imdb = fh_projection_utils.getDenslyProjectedImdb(angle_separated_imdb, 3, 0);
  experiments{end+1}.imdb = projected_imdb;
  experiments{end}.title = 'Dense RP = 3 - ReLU = 0';
  afprintf(sprintf('[INFO] done!\n'));

  afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  projected_imdb = fh_projection_utils.getDenslyProjectedImdb(angle_separated_imdb, 4, 0);
  experiments{end+1}.imdb = projected_imdb;
  experiments{end}.title = 'Dense RP = 4 - ReLU = 0';
  afprintf(sprintf('[INFO] done!\n'));

  afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  projected_imdb = fh_projection_utils.getDenslyProjectedImdb(angle_separated_imdb, 5, 0);
  experiments{end+1}.imdb = projected_imdb;
  experiments{end}.title = 'Dense RP = 5 - ReLU = 0';
  afprintf(sprintf('[INFO] done!\n'));

  afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  projected_imdb = fh_projection_utils.getDenslyProjectedImdb(angle_separated_imdb, 1, 1);
  experiments{end+1}.imdb = projected_imdb;
  experiments{end}.title = 'Dense RP = 1 - ReLU = 1';
  afprintf(sprintf('[INFO] done!\n'));

  afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  projected_imdb = fh_projection_utils.getDenslyProjectedImdb(angle_separated_imdb, 2, 2);
  experiments{end+1}.imdb = projected_imdb;
  experiments{end}.title = 'Dense RP = 2 - ReLU = 2';
  afprintf(sprintf('[INFO] done!\n'));

  afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  projected_imdb = fh_projection_utils.getDenslyProjectedImdb(angle_separated_imdb, 3, 3);
  experiments{end+1}.imdb = projected_imdb;
  experiments{end}.title = 'Dense RP = 3 - ReLU = 3';
  afprintf(sprintf('[INFO] done!\n'));

  afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  projected_imdb = fh_projection_utils.getDenslyProjectedImdb(angle_separated_imdb, 4, 4);
  experiments{end+1}.imdb = projected_imdb;
  experiments{end}.title = 'Dense RP = 4 - ReLU = 4';
  afprintf(sprintf('[INFO] done!\n'));

  afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  projected_imdb = fh_projection_utils.getDenslyProjectedImdb(angle_separated_imdb, 5, 5);
  experiments{end+1}.imdb = projected_imdb;
  experiments{end}.title = 'Dense RP = 5 - ReLU = 5';
  afprintf(sprintf('[INFO] done!\n'));


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV1P0RL0';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = larp_network_arch;
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV1P0RL1';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = larp_network_arch;
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV3P0RL0';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = larp_network_arch;
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV3P0RL3';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = larp_network_arch;
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV3P3RL0';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = larp_network_arch;
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV3P3RL3';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s (LeNet)', larp_network_arch);
  % afprintf(sprintf('[INFO] done!\n'));


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV8P0RL8';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, 2);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s-depth-%d', larp_network_arch, 2);
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV8P0RL8';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, 4);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s-depth-%d', larp_network_arch, 4);
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV8P0RL8';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, 6);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s-depth-%d', larp_network_arch, 6);
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV8P0RL8';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, 8);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s-depth-%d', larp_network_arch, 8);
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV8P0RL8';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, 10);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s-depth-%d', larp_network_arch, 10);
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV8P0RL8';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, 12);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s-depth-%d', larp_network_arch, 12);
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV8P0RL8';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, 14);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s-depth-%d', larp_network_arch, 14);
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV8P0RL8';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, 16);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s-depth-%d', larp_network_arch, 16);
  % afprintf(sprintf('[INFO] done!\n'));


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV5P0RL0';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = larp_network_arch;
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV5P0RL5';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = larp_network_arch;
  % afprintf(sprintf('[INFO] done!\n'));



  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV5P3RL0';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = larp_network_arch;
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV5P3RL5';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s (AlexNet)', larp_network_arch);
  % afprintf(sprintf('[INFO] done!\n'));



  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV5P5RL0';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = larp_network_arch;
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV5P5RL5';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = larp_network_arch;
  % afprintf(sprintf('[INFO] done!\n'));


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV1P0RL0', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV1P0RL0 - trained on 38';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV1P0RL1', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 2);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV1P0RL1 - trained on 38';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P0RL0', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 3);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV3P0RL0 - trained on 38';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P0RL3', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 6);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV3P0RL3 - trained on 38';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P3RL0', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 6);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV3P3RL0 - trained on 38';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P3RL3', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 9);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV3P3RL3 (LeNet) - trained on 38';
  % afprintf(sprintf('[INFO] done!\n'));



  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV1P0RL0', 'cifar', 'balanced-all');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV1P0RL0 - trained on all';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV1P0RL1', 'cifar', 'balanced-all');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 2);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV1P0RL1 - trained on all';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P0RL0', 'cifar', 'balanced-all');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 3);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV3P0RL0 - trained on all';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P0RL3', 'cifar', 'balanced-all');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 6);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV3P0RL3 - trained on all';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P3RL0', 'cifar', 'balanced-all');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 6);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV3P3RL0 - trained on all';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P3RL3', 'cifar', 'balanced-all');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 9);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV3P3RL3 (LeNet) - trained on all';
  % afprintf(sprintf('[INFO] done!\n'));


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % tmp = load(path_2);
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P3RL3', 'cifar', 'whatever');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 9);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV3P3RL3 - trained on ALL';
  % afprintf(sprintf('[INFO] done!\n'));



% -------------------------------------------------------------------------
function projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, projection_depth)
% -------------------------------------------------------------------------
  fh_projection_utils = projectionUtils;
  larp_weight_init_sequence = getLarpWeightInitSequence(larp_weight_init_type, larp_network_arch);
  projection_net = fh_projection_utils.getProjectionNetworkObject(dataset, larp_network_arch, larp_weight_init_sequence);
  projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, projection_depth);
