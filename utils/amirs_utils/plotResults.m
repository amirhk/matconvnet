% -------------------------------------------------------------------------
function plotResults(input_opts)
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

resultsDir = '/Volumes/Amir/results/';

epochNum = 50;
epochFile = sprintf('net-epoch-%d.mat', epochNum);

dataset = 'cifar';
networkArch = 'alexnet';

switch dataset
  case 'cifar'
    experimentDir = '2016-11-22-24; Varying Weight Initialization; CIFAR; LeNet; FC+{0-3}';
    experimentDir = '2016-11-04-06; Varying Weight Initialization; CIFAR; AlexNet; FC+{0-5} 222';
  case 'coil-100'
    experimentDir = '2016-11-22-24; Varying Weight Initialization; COIL-100; LeNet; FC+{0-3}';
  case 'mnist'
    experimentDir = '2016-11-22-24; Varying Weight Initialization; MNIST; LeNet; FC+{0-3}';
  case 'stl-10'
    experimentDir = '2016-11-22-24; Varying Weight Initialization; STL-10; LeNet; FC+{0-3}';
end

subExperimentDirs = { ....
  'compRand', ...
  ... '1_clustered_layerwise_1D_from_cifar', ...
  ... '2_clustered_layerwise_1D_from_cifar', ...
  ... '4_clustered_layerwise_1D_from_cifar', ...
  ... '8_clustered_layerwise_1D_from_cifar', ...
  ... '16_clustered_layerwise_1D_from_cifar', ...
  'kernelwise_1D_from_cifar', ...
  ... 'layerwise_1D_from_cifar', ...
  ... '1_clustered_layerwise_1D_from_coil_100', ...
  ... '2_clustered_layerwise_1D_from_coil_100', ...
  ... '4_clustered_layerwise_1D_from_coil_100', ...
  ... '8_clustered_layerwise_1D_from_coil_100', ...
  ... '16_clustered_layerwise_1D_from_coil_100', ...
  ... 'kernelwise_1D_from_coil_100', ...
  ... 'layerwise_1D_from_coil_100', ...
  ... '1_clustered_layerwise_1D_from_mnist', ...
  ... '2_clustered_layerwise_1D_from_mnist', ...
  ... '4_clustered_layerwise_1D_from_mnist', ...
  ... '8_clustered_layerwise_1D_from_mnist', ...
  ... '16_clustered_layerwise_1D_from_mnist', ...
  ... 'kernelwise_1D_from_mnist', ...
  ... 'layerwise_1D_from_mnist', ...
  ... '1_clustered_layerwise_1D_from_stl_10', ...
  ... '2_clustered_layerwise_1D_from_stl_10', ...
  ... '4_clustered_layerwise_1D_from_stl_10', ...
  ... '8_clustered_layerwise_1D_from_stl_10', ...
  ... '16_clustered_layerwise_1D_from_stl_10', ...
  ... 'kernelwise_1D_from_stl_10', ...
  ... 'layerwise_1D_from_stl_10', ...
};

fprintf('Loading files...\n'); i = 1;
fprintf('\nFinished loading result files!\n\n');
switch networkArch
  case 'alexnet'
    backPropDepthLimit = 5;
  case 'lenet'
    backPropDepthLimit = 3;
  case 'mnistnet'
    backPropDepthLimit = 2;
end


experiments = {};
for subExperimentDir = subExperimentDirs
  for j = 0:backPropDepthLimit
    bpd = getBPDForBPDLimit(j, backPropDepthLimit);
    subExperimentDir = char(subExperimentDir);
    experiments.(sprintf('exp_%s', subExperimentDir)).(sprintf('fc_plus_%d', j)) = loadEpochFile(resultsDir, experimentDir, subExperimentDir, bpd, epochFile); i = printLoadSuccess(i);
  end
end

startEpoch = 1;
granularity = 1;
for backPropDepth = 0:backPropDepthLimit
  for resultType = {'train', 'val'}
    resultType = char(resultType);

    % h = figure;
    % experiment = sprintf('Varying Weight Initialization - FC + %d', backPropDepth);
    % exp_1 = eval(sprintf('fc_plus_%d_3xkernelwise_1D', backPropDepth));
    % exp_2 = eval(sprintf('fc_plus_%d_3xcompRand', backPropDepth));
    % exp_3 = eval(sprintf('fc_plus_%d_3xlayerwise_1D_from_CIFAR', backPropDepth));
    % exp_4 = eval(sprintf('fc_plus_%d_3xlayerwise_1D_from_COIL_100', backPropDepth));
    % exp_5 = eval(sprintf('fc_plus_%d_3xlayerwise_1D_from_MNIST', backPropDepth));
    % exp_6 = eval(sprintf('fc_plus_%d_3xlayerwise_1D_from_STL_10', backPropDepth));
    % exp_7 = eval(sprintf('fc_plus_%d_3x4_clustered_layerwise_1D_from_CIFAR', backPropDepth));
    % exp_8 = eval(sprintf('fc_plus_%d_3x8_clustered_layerwise_1D_from_CIFAR', backPropDepth));
    % plot( ...
    %   startEpoch:1:epochNum, [exp_1.info.(resultType).error(1,startEpoch:epochNum)], 'r', ...
    %   startEpoch:1:epochNum, [exp_2.info.(resultType).error(1,startEpoch:epochNum)], 'k', ...
    %   startEpoch:1:epochNum, [exp_3.info.(resultType).error(1,startEpoch:epochNum)], 'r--', ...
    %   startEpoch:1:epochNum, [exp_4.info.(resultType).error(1,startEpoch:epochNum)], 'r-.', ...
    %   startEpoch:1:epochNum, [exp_5.info.(resultType).error(1,startEpoch:epochNum)], 'r-^', ...
    %   startEpoch:1:epochNum, [exp_6.info.(resultType).error(1,startEpoch:epochNum)], 'r:', ...
    %   startEpoch:1:epochNum, [exp_7.info.(resultType).error(1,startEpoch:epochNum)], 'y--', ...
    %   startEpoch:1:epochNum, [exp_8.info.(resultType).error(1,startEpoch:epochNum)], 'y:', ...
    %   'LineWidth', 2);
    % grid on
    % title(experiment);
    % legend(...
    %   '3 x kernelwise 1D', ...
    %   '3 x compRand', ...
    %   '3 x layerwise 1D from CIFAR', ...
    %   '3 x layerwise 1D from COIL-100', ...
    %   '3 x layerwise 1D from MNIST', ...
    %   '3 x layerwise 1D from STL-10', ...
    %   '3 x 4-clustered layerwise 1D from CIFAR', ...
    %   '3 x 8-clustered layerwise 1D from CIFAR');
    % xlabel('epoch')
    % % ylabel('Training Error');
    % ylim([0, 1 / granularity]);
    % switch resultType
    %   case 'train'
    %     ylabel('Training Error');
    %     fileName = sprintf('Training Comparison - %s.png', experiment);
    %   case 'val'
    %     ylabel('Validation Error');
    %     fileName = sprintf('Validation Comparison - %s.png', experiment);
    % end
    % saveas(h, fileName);


    % h = figure;
    % experiment_title = sprintf('Varying Weight Initialization - FC + %d', backPropDepth);
    % exp_1 = eval(sprintf('fc_plus_%d_3xkernelwise_1D', backPropDepth));
    % exp_2 = eval(sprintf('fc_plus_%d_3xcompRand', backPropDepth));
    % exp_3 = eval(sprintf('fc_plus_%d_3xlayerwise_1D_from_CIFAR', backPropDepth));
    % exp_4 = eval(sprintf('fc_plus_%d_3xlayerwise_1D_from_COIL_100', backPropDepth));
    % exp_5 = eval(sprintf('fc_plus_%d_3xlayerwise_1D_from_MNIST', backPropDepth));
    % exp_6 = eval(sprintf('fc_plus_%d_3xlayerwise_1D_from_STL_10', backPropDepth));
    % exp_7 = eval(sprintf('fc_plus_%d_3x4_clustered_layerwise_1D_from_CIFAR', backPropDepth));
    % exp_8 = eval(sprintf('fc_plus_%d_3x8_clustered_layerwise_1D_from_CIFAR', backPropDepth));
    % plot( ...
    %   startEpoch:1:epochNum, [exp_1.info.(resultType).error(1,startEpoch:epochNum)], 'r', ...
    %   startEpoch:1:epochNum, [exp_2.info.(resultType).error(1,startEpoch:epochNum)], 'k', ...
    %   startEpoch:1:epochNum, [exp_3.info.(resultType).error(1,startEpoch:epochNum)], 'r--', ...
    %   startEpoch:1:epochNum, [exp_4.info.(resultType).error(1,startEpoch:epochNum)], 'r-.', ...
    %   startEpoch:1:epochNum, [exp_5.info.(resultType).error(1,startEpoch:epochNum)], 'r-^', ...
    %   startEpoch:1:epochNum, [exp_6.info.(resultType).error(1,startEpoch:epochNum)], 'r:', ...
    %   startEpoch:1:epochNum, [exp_7.info.(resultType).error(1,startEpoch:epochNum)], 'y--', ...
    %   startEpoch:1:epochNum, [exp_8.info.(resultType).error(1,startEpoch:epochNum)], 'y:', ...
    %   'LineWidth', 2);

    h = figure;
    set(gca,'fontsize',18)
    hold on;
    grid on;
    experiment_title = sprintf('Varying Weight Initialization - FC + %d', backPropDepth);
    experiment_count = 0;
    legend_name_list = {};
    for subExperimentDir = subExperimentDirs
      experiment_count = experiment_count + 1;
      subExperimentDir = char(subExperimentDir);
      tmp_exp = experiments.(sprintf('exp_%s', subExperimentDir)).(sprintf('fc_plus_%d', backPropDepth));
      plot( ...
        startEpoch:1:epochNum, [tmp_exp.info.(resultType).error(1,startEpoch:epochNum)],  ...
        'LineWidth', 3);
      legend_name_list{experiment_count} = subExperimentDir;
      % fprintf('convergance accuracy %s: %6.5f\n', subExperimentDir, 1 - tmp_exp.info.(resultType).error(1,epochNum));
    end

    title(experiment_title);
    xlabel('epoch')
    ylim([0, 1 / granularity]);
    % legend(legend_name_list);
    legend(...
      'compRand', ...
      'kernelwise 1D', ...
      'layerwise 1D');
    switch resultType
      case 'train'
        ylabel('Training Error');
        % fileName = sprintf('Training Comparison - %s.png', experiment_title);
        fileName = sprintf('Training Comparison - %s.eps', experiment_title);
      case 'val'
        ylabel('Validation Error');
        % fileName = sprintf('Validation Comparison - %s.png', experiment_title);
        fileName = sprintf('Validation Comparison - %s.eps', experiment_title);
        fileName = sprintf('fc_%d', backPropDepth);
    end
    % saveas(h, fileName);
    saveas(h, fileName, 'epsc');










    fprintf('\n-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- \n\n');
    fprintf('Results for:\n');
    fprintf('\tFC + %d\n', backPropDepth);
    fprintf('\tType: %s\n\n', resultType);

    for subExperimentDir = subExperimentDirs
      subExperimentDir = char(subExperimentDir);
      tmp_exp = experiments.(sprintf('exp_%s', subExperimentDir)).(sprintf('fc_plus_%d', backPropDepth));
      fprintf('convergance accuracy %s: %6.5f\n', subExperimentDir, 1 - tmp_exp.info.(resultType).error(1,epochNum));
    end
  end
end

% --------------------------------------------------------------------
function subDirName = findSubDirWithBackPropDepth(parent_dir, input_bpd)
% --------------------------------------------------------------------
  parent_dir_obj = dir(parent_dir);
  for i = 1:length(parent_dir_obj)
    tmpSubDirName = parent_dir_obj(i).name;
    index_of_bpd_string = strfind(tmpSubDirName, 'bpd');
    if index_of_bpd_string
      % examples:
      %   some-experiment-name-gpu-1-bpd-7
      %   some-experiment-name-gpu-1-bpd-13
      bpd = tmpSubDirName(index_of_bpd_string + 4:end);
      if str2num(bpd) == input_bpd
        subDirName = tmpSubDirName;
        return
      end
    end
  end
end

% --------------------------------------------------------------------
function loadedEpochFile = loadEpochFile(resultsDir, experimentDir, subExperimentDir, bpd, epochFile);
% --------------------------------------------------------------------
  loadedEpochFile = load(fullfile( ...
    resultsDir, ...
    experimentDir, ...
    subExperimentDir, ...
    findSubDirWithBackPropDepth(fullfile(resultsDir, experimentDir, subExperimentDir), bpd), ...
    epochFile));
end

% --------------------------------------------------------------------
function bpd = getBPDForBPDLimit(index, bpd_bucket_count);
% --------------------------------------------------------------------
  switch bpd_bucket_count
    case 3
      switch index
        case 0
          bpd = 4;
        case 1
          bpd = 7;
        case 2
          bpd = 10;
        case 3
          bpd = 13;
      end
    case 5
      switch index
        case 0
          bpd = 7;
        case 1
          bpd = 10;
        case 2
          bpd = 12;
        case 3
          bpd = 15;
        case 4
          bpd = 18;
        case 5
          bpd = 20;
      end
  end
end

% --------------------------------------------------------------------
function i = printLoadSuccess(i);
% --------------------------------------------------------------------
  fprintf('\t[INFO] file %d loaded successfully.\n', i);
  i = i + 1;
end
