resultsDir = '/Volumes/Amir/results/';

epochNum = 50;
epochFile = sprintf('net-epoch-%d.mat', epochNum);

dataset = 'stl-10';
networkArch = 'lenet';

switch dataset
  case 'cifar'
    experimentDir = '2016-11-22-24; Varying Weight Initialization; CIFAR; LeNet; FC+{0-3}';
  case 'coil-100'
    experimentDir = '2016-11-22-24; Varying Weight Initialization; COIL-100; LeNet; FC+{0-3}';
  case 'mnist'
    experimentDir = '2016-11-22-24; Varying Weight Initialization; MNIST; LeNet; FC+{0-3}';
  case 'stl-10'
    experimentDir = '2016-11-22-24; Varying Weight Initialization; STL-10; LeNet; FC+{0-3}';
end

subExperimentDirs = { ....
  'compRand', ...
  '1_clustered_layerwise_1D_from_cifar', ...
  '2_clustered_layerwise_1D_from_cifar', ...
  '4_clustered_layerwise_1D_from_cifar', ...
  '8_clustered_layerwise_1D_from_cifar', ...
  '16_clustered_layerwise_1D_from_cifar', ...
  'kernelwise_1D_from_cifar', ...
  'layerwise_1D_from_cifar', ...
  '1_clustered_layerwise_1D_from_coil_100', ...
  '2_clustered_layerwise_1D_from_coil_100', ...
  '4_clustered_layerwise_1D_from_coil_100', ...
  '8_clustered_layerwise_1D_from_coil_100', ...
  '16_clustered_layerwise_1D_from_coil_100', ...
  'kernelwise_1D_from_coil_100', ...
  'layerwise_1D_from_coil_100', ...
  '1_clustered_layerwise_1D_from_mnist', ...
  '2_clustered_layerwise_1D_from_mnist', ...
  '4_clustered_layerwise_1D_from_mnist', ...
  '8_clustered_layerwise_1D_from_mnist', ...
  '16_clustered_layerwise_1D_from_mnist', ...
  'kernelwise_1D_from_mnist', ...
  'layerwise_1D_from_mnist', ...
  '1_clustered_layerwise_1D_from_stl_10', ...
  '2_clustered_layerwise_1D_from_stl_10', ...
  '4_clustered_layerwise_1D_from_stl_10', ...
  '8_clustered_layerwise_1D_from_stl_10', ...
  '16_clustered_layerwise_1D_from_stl_10', ...
  'kernelwise_1D_from_stl_10', ...
  'layerwise_1D_from_stl_10', ...
};

fprintf('Loading files...\n'); i = 1;

experiments = {};
for subExperimentDir = subExperimentDirs
  subExperimentDir = char(subExperimentDir);
  experiments.(sprintf('exp_%s', subExperimentDir)).fc_plus_3 = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13, epochFile); i = printLoadSuccess(i);
  experiments.(sprintf('exp_%s', subExperimentDir)).fc_plus_2 = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 10, epochFile); i = printLoadSuccess(i);
  experiments.(sprintf('exp_%s', subExperimentDir)).fc_plus_1 = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 7, epochFile); i = printLoadSuccess(i);
  experiments.(sprintf('exp_%s', subExperimentDir)).fc_plus_0 = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 4, epochFile); i = printLoadSuccess(i);
end

fprintf('\nFinished loading result files!\n\n');
switch networkArch
  case 'alexnet'
    backPropDepthLimit = 5;
  case 'lenet'
    backPropDepthLimit = 3;
  case 'mnistnet'
    backPropDepthLimit = 2;
end

startEpoch = 1;
% granularity = 1;
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

    fprintf('\n-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- \n\n');
    fprintf('Results for:\n');
    fprintf('\tFC + %d\n', backPropDepth);
    fprintf('\tType: %s\n\n', resultType);

    for subExperimentDir = subExperimentDirs
      subExperimentDir = char(subExperimentDir);
      tmp_exp = experiments.(sprintf('exp_%s', subExperimentDir)).(sprintf('fc_plus_%d', backPropDepth));
      fprintf('convergance accuracy %s: %6.5f\n', subExperimentDir, 1 - tmp_exp.info.(resultType).error(1,epochNum));
    end

    % exp_01 = experiments.exp_compRand.(sprintf('fc_plus_%d', backPropDepth));
    % exp_02 = experiments.exp_1_clustered_layerwise_1D_from_cifar.(sprintf('fc_plus_%d', backPropDepth));
    % exp_03 = experiments.exp_2_clustered_layerwise_1D_from_cifar.(sprintf('fc_plus_%d', backPropDepth));
    % exp_04 = experiments.exp_4_clustered_layerwise_1D_from_cifar.(sprintf('fc_plus_%d', backPropDepth));
    % exp_05 = experiments.exp_8_clustered_layerwise_1D_from_cifar.(sprintf('fc_plus_%d', backPropDepth));
    % exp_06 = experiments.exp_16_clustered_layerwise_1D_from_cifar.(sprintf('fc_plus_%d', backPropDepth));
    % exp_07 = experiments.exp_kernelwise_1D_from_cifar.(sprintf('fc_plus_%d', backPropDepth));
    % exp_08 = experiments.exp_layerwise_1D_from_cifar.(sprintf('fc_plus_%d', backPropDepth));
    % exp_09 = experiments.exp_1_clustered_layerwise_1D_from_coil_100.(sprintf('fc_plus_%d', backPropDepth));
    % exp_10 = experiments.exp_2_clustered_layerwise_1D_from_coil_100.(sprintf('fc_plus_%d', backPropDepth));
    % exp_11 = experiments.exp_4_clustered_layerwise_1D_from_coil_100.(sprintf('fc_plus_%d', backPropDepth));
    % exp_12 = experiments.exp_8_clustered_layerwise_1D_from_coil_100.(sprintf('fc_plus_%d', backPropDepth));
    % exp_13 = experiments.exp_16_clustered_layerwise_1D_from_coil_100.(sprintf('fc_plus_%d', backPropDepth));
    % exp_14 = experiments.exp_kernelwise_1D_from_coil_100.(sprintf('fc_plus_%d', backPropDepth));
    % exp_15 = experiments.exp_layerwise_1D_from_coil_100.(sprintf('fc_plus_%d', backPropDepth));
    % exp_16 = experiments.exp_1_clustered_layerwise_1D_from_mnist.(sprintf('fc_plus_%d', backPropDepth));
    % exp_17 = experiments.exp_2_clustered_layerwise_1D_from_mnist.(sprintf('fc_plus_%d', backPropDepth));
    % exp_18 = experiments.exp_4_clustered_layerwise_1D_from_mnist.(sprintf('fc_plus_%d', backPropDepth));
    % exp_19 = experiments.exp_8_clustered_layerwise_1D_from_mnist.(sprintf('fc_plus_%d', backPropDepth));
    % exp_20 = experiments.exp_16_clustered_layerwise_1D_from_mnist.(sprintf('fc_plus_%d', backPropDepth));
    % exp_21 = experiments.exp_kernelwise_1D_from_mnist.(sprintf('fc_plus_%d', backPropDepth));
    % exp_22 = experiments.exp_layerwise_1D_from_mnist.(sprintf('fc_plus_%d', backPropDepth));
    % exp_23 = experiments.exp_1_clustered_layerwise_1D_from_stl_10.(sprintf('fc_plus_%d', backPropDepth));
    % exp_24 = experiments.exp_2_clustered_layerwise_1D_from_stl_10.(sprintf('fc_plus_%d', backPropDepth));
    % exp_25 = experiments.exp_4_clustered_layerwise_1D_from_stl_10.(sprintf('fc_plus_%d', backPropDepth));
    % exp_26 = experiments.exp_8_clustered_layerwise_1D_from_stl_10.(sprintf('fc_plus_%d', backPropDepth));
    % exp_27 = experiments.exp_16_clustered_layerwise_1D_from_stl_10.(sprintf('fc_plus_%d', backPropDepth));
    % exp_28 = experiments.exp_kernelwise_1D_from_stl_10.(sprintf('fc_plus_%d', backPropDepth));
    % exp_29 = experiments.exp_layerwise_1D_from_stl_10.(sprintf('fc_plus_%d', backPropDepth));

    % fprintf('\n-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- \n\n');
    % fprintf('Results for:\n');
    % fprintf('\tFC + %d\n', backPropDepth);
    % fprintf('\tType: %s\n\n', resultType);

    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_01.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_02.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_03.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_04.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_05.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_06.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_07.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_08.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_09.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_10.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_11.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_12.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_13.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_14.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_15.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_16.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_17.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_18.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_19.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_20.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_21.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_22.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_23.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_24.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_25.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_26.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_27.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_28.info.(resultType).error(1,epochNum));
    % fprintf('convergance accuracy: %6.5f\n', 1 - exp_29.info.(resultType).error(1,epochNum));
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
function i = printLoadSuccess(i);
% --------------------------------------------------------------------
  fprintf('\t[INFO] file %d loaded successfully.\n', i);
  i = i + 1;
end
