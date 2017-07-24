% -------------------------------------------------------------------------
function [original_imdb, experiments] = setupExperimentsUsingProjectedImbds(dataset, posneg_balance, should_filter_out_test_set, debug_flag)
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

  if debug_flag, afprintf(sprintf('[INFO] Loading original imdb...\n')); end;
  tmp_opts.dataset = dataset;
  tmp_opts.posneg_balance = posneg_balance;
  original_imdb = loadSavedImdb(tmp_opts, debug_flag);
  if should_filter_out_test_set
    original_imdb = filterImdbForSet(original_imdb, 1, 1);
  end
  if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;





  % projection = 'dense_rp';
  % % projection = 'dense_rp_normalized';
  % % projection = 'dense_lognormal';
  % switch projection
  %   case 'dense_rp'
  %     fhGetDenslyProjectedImdb = fh_projection_utils.getDenslyProjectedImdb;
  %     title_prefix = 'Dense RP';
  %   case 'dense_rp_normalized'
  %     fhGetDenslyProjectedImdb = fh_projection_utils.getDenslyProjectedAndNormalizedImdb;
  %     title_prefix = 'Dense RP Normalized';
  %   case 'dense_lognormal'
  %     fhGetDenslyProjectedImdb = fh_projection_utils.getDenslyLogNormalProjectedImdb;
  %     title_prefix = 'LogNormal Dense RP';
  % end
  % % if false % TODO... this doesn't actaully cahnge the file name!!!!!!!
  % %   fhGetDenslyProjectedImdb = fh_projection_utils.getDenslyProjectedImdb;
  % %   title_prefix = 'Dense RP';
  % % else
  % %   fhGetDenslyProjectedImdb = fh_projection_utils.getDenslyLogNormalProjectedImdb;
  % %   title_prefix = 'LogNormal Dense RP';
  % % end


  % % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  % experiments{end+1}.imdb = original_imdb;
  % experiments{end}.title = 'Original IMDB';


  % % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projected_imdb = fhGetDenslyProjectedImdb(original_imdb, 1, 0);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s = 1 - ReLU = 0', title_prefix);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projected_imdb = fhGetDenslyProjectedImdb(original_imdb, 2, 0);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s = 2 - ReLU = 0', title_prefix);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projected_imdb = fhGetDenslyProjectedImdb(original_imdb, 3, 0);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s = 3 - ReLU = 0', title_prefix);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projected_imdb = fhGetDenslyProjectedImdb(original_imdb, 4, 0);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s = 4 - ReLU = 0', title_prefix);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projected_imdb = fhGetDenslyProjectedImdb(original_imdb, 5, 0);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s = 5 - ReLU = 0', title_prefix);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projected_imdb = fhGetDenslyProjectedImdb(original_imdb, 1, 1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s = 1 - ReLU = 1', title_prefix);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projected_imdb = fhGetDenslyProjectedImdb(original_imdb, 2, 2);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s = 2 - ReLU = 2', title_prefix);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projected_imdb = fhGetDenslyProjectedImdb(original_imdb, 3, 3);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s = 3 - ReLU = 3', title_prefix);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projected_imdb = fhGetDenslyProjectedImdb(original_imdb, 4, 4);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s = 4 - ReLU = 4', title_prefix);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projected_imdb = fhGetDenslyProjectedImdb(original_imdb, 5, 5);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s = 5 - ReLU = 5', title_prefix);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % angle_separated_imdb = fh_projection_utils.getAngleSeparatedImdb(original_imdb);
  % experiments{end+1}.imdb = angle_separated_imdb;
  % experiments{end}.title = 'Angle Separated Imdb';
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projected_imdb = fhGetDenslyProjectedImdb(angle_separated_imdb, 1, 0);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s = 1 - ReLU = 0', title_prefix);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projected_imdb = fhGetDenslyProjectedImdb(angle_separated_imdb, 2, 0);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s = 2 - ReLU = 0', title_prefix);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projected_imdb = fhGetDenslyProjectedImdb(angle_separated_imdb, 3, 0);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s = 3 - ReLU = 0', title_prefix);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projected_imdb = fhGetDenslyProjectedImdb(angle_separated_imdb, 4, 0);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s = 4 - ReLU = 0', title_prefix);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projected_imdb = fhGetDenslyProjectedImdb(angle_separated_imdb, 5, 0);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s = 5 - ReLU = 0', title_prefix);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projected_imdb = fhGetDenslyProjectedImdb(angle_separated_imdb, 1, 1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s = 1 - ReLU = 1', title_prefix);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projected_imdb = fhGetDenslyProjectedImdb(angle_separated_imdb, 2, 2);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s = 2 - ReLU = 2', title_prefix);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projected_imdb = fhGetDenslyProjectedImdb(angle_separated_imdb, 3, 3);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s = 3 - ReLU = 3', title_prefix);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projected_imdb = fhGetDenslyProjectedImdb(angle_separated_imdb, 4, 4);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s = 4 - ReLU = 4', title_prefix);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projected_imdb = fhGetDenslyProjectedImdb(angle_separated_imdb, 5, 5);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s = 5 - ReLU = 5', title_prefix);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --



































% -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- --
% -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- --
% -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- --
% -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- --








  % experiments{end+1}.imdb = original_imdb;
  % experiments{end}.title = sprintf('%s - %s - Original IMDB', dataset, posneg_balance);











  % % % projected_dim_list = [4, 16, 64, 256, 1024, 4096, 16384, 65536];
  % % % projected_dim_list = [4, 16, 64, 256, 1024, 4096, 16384];
  % projected_dim_list = [4, 16, 64, 256, 1024, 4096];
  % % % projected_dim_list = [4, 16, 64];
  % % projected_dim_list = [16384];

  % for projected_dim = projected_dim_list

  %   if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  %   projection_description = sprintf('larpD1P0RL0 w/ dense_gaussian into %d', projected_dim);
  %   projected_imdb = fh_projection_utils.getDenslyDownProjectedImdb(original_imdb, 1, 'dense_gaussian', 0, 'relu', projected_dim);
  %   experiments{end+1}.imdb = projected_imdb;
  %   experiments{end}.title = sprintf('%s - %s - projected through: %s', dataset, posneg_balance, projection_description);
  %   if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % end



  % for projected_dim = projected_dim_list

  %   if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  %   projection_description = sprintf('larpD1P0RL1 w/ dense_gaussian into %d', projected_dim);
  %   projected_imdb = fh_projection_utils.getDenslyDownProjectedImdb(original_imdb, 1, 'dense_gaussian', 1, 'relu', projected_dim);
  %   experiments{end+1}.imdb = projected_imdb;
  %   experiments{end}.title = sprintf('%s - %s - projected through: %s', dataset, posneg_balance, projection_description);
  %   if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % end

















  weight_distribution_list = { ...,
    'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1', ...
    'logNormal-layer5-ratVisualCortex', ...
    'gaussian-CentreSurroundCovariance-randomDivide-10-MuDivide-1-SigmaDivide-1'};

  % larp_network_arch_list = { ...
  %   'larpV1P1RL1-special-pooling-1', ...     % 256
  %   'larpV1P1RL1-special-pooling-2', ...     % 256
  %   'larpV3P3RL3-final-conv-16-kernels', ... % 256 (change # filters in final layer in getLarpArch.m)
  %   'larpV5P3RL5-final-conv-16-kernels', ... % 256 (change # filters in final layer in getLarpArch.m)
  %   'larpV3P3RL3', ...                       % 1,024
  %   'larpV5P3RL5', ...                       % 1,024
  %   'larpV3P2RL3', ...                       % 4,096
  %   'larpV5P2RL5', ...                       % 4,096
  %   'larpV1P1RL1', ...                       % 16,384
  %   'larpV3P1RL3', ...                       % 16,384
  %   'larpV5P1RL5', ...                       % 16,384
  %   'larpV1P1RL1-non-decimated-pooling', ... % 65,536
  %   'larpV3P1RL3-non-decimated-pooling', ... % 65,536
  %   'larpV5P1RL5-non-decimated-pooling', ... % 65,536
  %   'larpV1P0RL1', ...                       % 65,536
  %   'larpV3P0RL3', ...                       % 65,536
  %   'larpV5P0RL5'};                          % 65,536

  larp_network_arch_list = { ...
    'larpV1P1RL1-special-pooling-1', ...
    'larpV1P1RL1-special-pooling-2'};


  for larp_network_arch = larp_network_arch_list
    larp_network_arch = char(larp_network_arch);

    for weight_distribution = weight_distribution_list
      larp_weight_init_type = char(weight_distribution);

      if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
      projection_description = sprintf('%s w/ %s', larp_network_arch, larp_weight_init_type);
      projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
      experiments{end+1}.imdb = projected_imdb;
      experiments{end}.title = sprintf('%s - %s - projected through: %s', dataset, posneg_balance, projection_description);
      if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

    end

  end








  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projection_net = loadTrainedNet('larpV0P0RL0+convV1P1RL1', 'cifar', 'balanced-50');
  % projection_description = 'jigar larpV0P0RL0+convV1P1RL1 trained on cifar balanced-50';
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 3);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s - %s - projected through: %s', dataset, posneg_balance, projection_description);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;


  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P1RL3', 'cifar', 'balanced-50');
  % projection_description = 'jigar larpV0P0RL0+convV3P1RL3 trained on cifar balanced-50';
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 3);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s - %s - projected through: %s', dataset, posneg_balance, projection_description);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;


  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projection_net = loadTrainedNet('larpV0P0RL0+convV5P1RL5', 'cifar', 'balanced-50');
  % projection_description = 'jigar larpV0P0RL0+convV5P1RL5 trained on cifar balanced-50';
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 3);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s - %s - projected through: %s', dataset, posneg_balance, projection_description);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;


  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projection_net = loadTrainedNet('larpV0P0RL0+convV1P0RL1', 'cifar', 'balanced-50');
  % projection_description = 'jigar larpV0P0RL0+convV1P0RL1 trained on cifar balanced-50';
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 3);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s - %s - projected through: %s', dataset, posneg_balance, projection_description);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;


  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P0RL3', 'cifar', 'balanced-50');
  % projection_description = 'jigar larpV0P0RL0+convV3P0RL3 trained on cifar balanced-50';
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 3);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s - %s - projected through: %s', dataset, posneg_balance, projection_description);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;


  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projection_net = loadTrainedNet('larpV0P0RL0+convV5P0RL5', 'cifar', 'balanced-50');
  % projection_description = 'jigar larpV0P0RL0+convV5P0RL5 trained on cifar balanced-50';
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 3);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s - %s - projected through: %s', dataset, posneg_balance, projection_description);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;













 % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- --
 % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- --
 % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- --
 % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- --
 % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- -- % -- -- --















































































  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV1P0RL0';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = larp_network_arch;
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV1P0RL1';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = larp_network_arch;
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV3P0RL0';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = larp_network_arch;
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV3P0RL3';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = larp_network_arch;
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV3P3RL0';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = larp_network_arch;
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV3P3RL3';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s (LeNet)', larp_network_arch);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV8P0RL8';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, 2);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s-depth-%d', larp_network_arch, 2);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV8P0RL8';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, 4);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s-depth-%d', larp_network_arch, 4);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV8P0RL8';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, 6);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s-depth-%d', larp_network_arch, 6);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV8P0RL8';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, 8);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s-depth-%d', larp_network_arch, 8);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV8P0RL8';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, 10);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s-depth-%d', larp_network_arch, 10);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV8P0RL8';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, 12);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s-depth-%d', larp_network_arch, 12);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV8P0RL8';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, 14);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s-depth-%d', larp_network_arch, 14);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV8P0RL8';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, 16);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s-depth-%d', larp_network_arch, 16);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV5P0RL0';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = larp_network_arch;
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV5P0RL5';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = larp_network_arch;
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;



  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV5P3RL0';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = larp_network_arch;
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV5P3RL5';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = sprintf('%s (AlexNet)', larp_network_arch);
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;



  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV5P5RL0';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = larp_network_arch;
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV5P5RL5';
  % projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = larp_network_arch;
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projection_net = loadTrainedNet('larpV0P0RL0+convV1P0RL0', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV1P0RL0 - trained on 38';
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projection_net = loadTrainedNet('larpV0P0RL0+convV1P0RL1', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 2);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV1P0RL1 - trained on 38';
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P0RL0', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 3);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV3P0RL0 - trained on 38';
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P0RL3', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 6);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV3P0RL3 - trained on 38';
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P3RL0', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 6);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV3P3RL0 - trained on 38';
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P3RL3', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 9);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV3P3RL3 (LeNet) - trained on 38';
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;



  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projection_net = loadTrainedNet('larpV0P0RL0+convV1P0RL0', 'cifar', 'balanced-all');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV1P0RL0 - trained on all';
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projection_net = loadTrainedNet('larpV0P0RL0+convV1P0RL1', 'cifar', 'balanced-all');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 2);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV1P0RL1 - trained on all';
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P0RL0', 'cifar', 'balanced-all');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 3);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV3P0RL0 - trained on all';
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P0RL3', 'cifar', 'balanced-all');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 6);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV3P0RL3 - trained on all';
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P3RL0', 'cifar', 'balanced-all');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 6);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV3P3RL0 - trained on all';
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;

  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P3RL3', 'cifar', 'balanced-all');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 9);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV3P3RL3 (LeNet) - trained on all';
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  % if debug_flag, afprintf(sprintf('[INFO] Loading projected imdb...\n')); end;
  % tmp = load(path_2);
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P3RL3', 'cifar', 'whatever');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 9);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained larpV0P0RL0+convV3P3RL3 - trained on ALL';
  % if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;



% -------------------------------------------------------------------------
function projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, projection_depth)
% -------------------------------------------------------------------------
  fh_projection_utils = projectionUtils;
  larp_weight_init_sequence = getLarpWeightInitSequence(larp_weight_init_type, larp_network_arch);
  projection_net = fh_projection_utils.getProjectionNetworkObject(dataset, larp_network_arch, larp_weight_init_sequence);
  projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, projection_depth);
