function main_cnn_amir(varargin)
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                   -- ==
% -- ==                        NETWORK ARCH                               -- ==
% -- ==                                                                   -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
  % dataset_list = {'cifar', 'stl-10', 'coil-100'}; % {'mnist', 'cifar', 'stl-10', 'coil-100'}
  % dataset_list = {'cifar', 'coil-100', 'mnist', 'stl-10'};
  dataset_list = {'mnist-two-class-unbalanced'};

  % network_arch = 'mnistnet';
  % % backprop_depth_list = [8, 6, 4];
  % backprop_depth_list = [4];

  % network_arch = 'prostatenet';
  % % backprop_depth_list = [13, 10, 7, 4];
  % % backprop_depth_list = [13, 4];
  % backprop_depth_list = [4];
  % % leave_out_type = 'sample';
  % leave_out_indices = 1:1:266;
  leave_out_type = 'patient';
  % leave_out_indices = 1:1:104;
  leave_out_indices = 1:1:1;


  network_arch = 'lenet';
  % backprop_depth_list = [13, 10, 7, 4]; % no dropout
  % backprop_depth_list = [14, 10, 7, 4]; % 1 x dropout after 1st layer
  % backprop_depth_list = [14, 11, 8, 4]; % 1 x dropout after 3rd layer
  % backprop_depth_list = [14, 11, 8, 5]; % 1 x dropout in FC
  % backprop_depth_list = [15, 11, 8, 4]; % 2 x dropout after 1st and 3rd layers
  % backprop_depth_list = [13];
  backprop_depth_list = [4];

  % network_arch = 'lenet';
  % backprop_depth_list = [13, 10, 7, 4];
  % backprop_depth_list = [4];

  % network_arch = 'alexnet';
  % % backprop_depth_list = [20, 18, 15, 12, 10, 7];
  % backprop_depth_list = [20];

  % network_arch = 'alexnet-bnorm';
  % % backprop_depth_list = [20, 18, 15, 12, 10, 7];
  % backprop_depth_list = [22];

  % network_arch = 'alexnet-bottleneck';
  % backprop_depth_list = [21];
  % bottleneck_divide_by_list = [1,2,4,8,16,32];
  bottleneck_divide_by_list = [1];

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                   -- ==
% -- ==                          MORE PARAMS                              -- ==
% -- ==                                                                   -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

  weight_init_source = 'gen';  % {'load' | 'gen'}
  weight_init_sequence_list = {{'compRand', 'compRand', 'compRand'}};

  % weight_init_source = 'load';  % {'load' | 'gen'}
  % % weight_init_sequence_list = {{'compRand', 'layerwise-1D-from-cifar', 'layerwise-1D-from-cifar'}};
  % weight_init_sequence_list = { ...
  %   ... % {'1-clustered-layerwise-1D-from-cifar', '1-clustered-layerwise-1D-from-cifar', '1-clustered-layerwise-1D-from-cifar'}, ...
  %   ... % {'1-clustered-layerwise-1D-from-coil-100', '1-clustered-layerwise-1D-from-coil-100', '1-clustered-layerwise-1D-from-coil-100'}, ...
  %   ... % {'1-clustered-layerwise-1D-from-mnist', '1-clustered-layerwise-1D-from-mnist', '1-clustered-layerwise-1D-from-mnist'}, ...
  %   ... % {'1-clustered-layerwise-1D-from-stl-10', '1-clustered-layerwise-1D-from-stl-10', '1-clustered-layerwise-1D-from-stl-10'}, ...
  %   ... % {'2-clustered-layerwise-1D-from-cifar', '2-clustered-layerwise-1D-from-cifar', '2-clustered-layerwise-1D-from-cifar'}, ...
  %   ... % {'2-clustered-layerwise-1D-from-coil-100', '2-clustered-layerwise-1D-from-coil-100', '2-clustered-layerwise-1D-from-coil-100'}, ...
  %   ... % {'2-clustered-layerwise-1D-from-mnist', '2-clustered-layerwise-1D-from-mnist', '2-clustered-layerwise-1D-from-mnist'}, ...
  %   ... % {'2-clustered-layerwise-1D-from-stl-10', '2-clustered-layerwise-1D-from-stl-10', '2-clustered-layerwise-1D-from-stl-10'}, ...
  %   ... % {'4-clustered-layerwise-1D-from-cifar', '4-clustered-layerwise-1D-from-cifar', '4-clustered-layerwise-1D-from-cifar'}, ...
  %   ... % {'4-clustered-layerwise-1D-from-coil-100', '4-clustered-layerwise-1D-from-coil-100', '4-clustered-layerwise-1D-from-coil-100'}, ...
  %   ... % {'4-clustered-layerwise-1D-from-mnist', '4-clustered-layerwise-1D-from-mnist', '4-clustered-layerwise-1D-from-mnist'}, ...
  %   ... % {'4-clustered-layerwise-1D-from-stl-10', '4-clustered-layerwise-1D-from-stl-10', '4-clustered-layerwise-1D-from-stl-10'}, ...
  %   ... % {'8-clustered-layerwise-1D-from-cifar', '8-clustered-layerwise-1D-from-cifar', '8-clustered-layerwise-1D-from-cifar'}, ...
  %   ... % {'8-clustered-layerwise-1D-from-coil-100', '8-clustered-layerwise-1D-from-coil-100', '8-clustered-layerwise-1D-from-coil-100'}, ...
  %   ... % {'8-clustered-layerwise-1D-from-mnist', '8-clustered-layerwise-1D-from-mnist', '8-clustered-layerwise-1D-from-mnist'}, ...
  %   ... % {'8-clustered-layerwise-1D-from-stl-10', '8-clustered-layerwise-1D-from-stl-10', '8-clustered-layerwise-1D-from-stl-10'}, ...
  %   ... % {'16-clustered-layerwise-1D-from-cifar', '16-clustered-layerwise-1D-from-cifar', '16-clustered-layerwise-1D-from-cifar'}, ...
  %   ... % {'16-clustered-layerwise-1D-from-coil-100', '16-clustered-layerwise-1D-from-coil-100', '16-clustered-layerwise-1D-from-coil-100'}, ...
  %   ... % {'16-clustered-layerwise-1D-from-mnist', '16-clustered-layerwise-1D-from-mnist', '16-clustered-layerwise-1D-from-mnist'}, ...
  %   ... % {'16-clustered-layerwise-1D-from-stl-10', '16-clustered-layerwise-1D-from-stl-10', '16-clustered-layerwise-1D-from-stl-10'}, ...
  %   {'compRand', 'compRand', 'compRand'}, ...
  %   {'kernelwise-1D-from-cifar', 'kernelwise-1D-from-cifar', 'kernelwise-1D-from-cifar'}, ...
  %   ... % {'kernelwise-1D-from-coil-100', 'kernelwise-1D-from-coil-100', 'kernelwise-1D-from-coil-100'}, ...
  %   ... % {'kernelwise-1D-from-mnist', 'kernelwise-1D-from-mnist', 'kernelwise-1D-from-mnist'}, ...
  %   ... % {'kernelwise-1D-from-stl-10', 'kernelwise-1D-from-stl-10', 'kernelwise-1D-from-stl-10'}, ...
  %   {'layerwise-1D-from-cifar', 'layerwise-1D-from-cifar', 'layerwise-1D-from-cifar'}, ...
  %   ... % {'layerwise-1D-from-coil-100', 'layerwise-1D-from-coil-100', 'layerwise-1D-from-coil-100'}, ...
  %   ... % {'layerwise-1D-from-mnist', 'layerwise-1D-from-mnist', 'layerwise-1D-from-mnist'}, ...
  %   ... % {'layerwise-1D-from-stl-10', 'layerwise-1D-from-stl-10', 'layerwise-1D-from-stl-10'}, ...
  % };

  % imdb_portion_list = [0.1, 0.25, 0.5, 1.0];
  imdb_portion_list = [1.0];

  % weight_decay_list = [0.1, 0.01, 0.001, 0.0001, 0]; % Works: {0.001, 0.0001, 0} Doesn't Work: {0.1, 0.01}
  weight_decay_list = [0.0001];

  debug_flag = true;

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                   -- ==
% -- ==                           MAIN LOOP                               -- ==
% -- ==                                                                   -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
  for dataset = dataset_list
    for weight_init_sequence = weight_init_sequence_list
      for bottleneck_divide_by = bottleneck_divide_by_list
        for weight_decay = weight_decay_list
          for backprop_depth = backprop_depth_list
            for leave_out_index = leave_out_indices
              tmp = load(fullfile(getDevPath(), 'data', 'saved-two-class-mnist.mat'));
              opts.imdb = tmp.imdb;
              opts.dataset = char(dataset);
              opts.network_arch = network_arch;
              opts.backprop_depth = backprop_depth;
              opts.weight_decay = weight_decay;
              opts.weight_init_sequence = weight_init_sequence{1};
              opts.weight_init_source = weight_init_source;
              opts.bottleneck_divide_by = bottleneck_divide_by;
              opts.leave_out_type = leave_out_type;
              opts.leave_out_index = leave_out_index;
              opts.debug_flag = debug_flag;
              opts.regen = true;
              cnnAmir(opts);
            end
          end
        end
      end
    end
  end
