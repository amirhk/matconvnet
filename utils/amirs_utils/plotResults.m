resultsDir = '/Volumes/Amir/results/';

epochNum = 50;
epochFile = sprintf('net-epoch-%d.mat', epochNum);
fprintf('Loading files...'); i = 1;

dataset = 'stl-10';
networkArch = 'lenet';

switch dataset
  case 'cifar'
    experimentDir = '2016-11-18-21; Varying Layerwise Weight Initialization; CIFAR; LeNet; FC+{0-3}';
  case 'coil-100'
    experimentDir = '2016-11-18-21; Varying Layerwise Weight Initialization; COIL-100; LeNet; FC+{0-3}';
  case 'mnist'
    experimentDir = '2016-11-18-21; Varying Layerwise Weight Initialization; MNIST; LeNet; FC+{0-3}';
  case 'stl-10'
    experimentDir = '2016-11-18-21; Varying Layerwise Weight Initialization; STL-10; LeNet; FC+{0-3}';
end

subExperimentDir = 'kernelwise-1D';
fc_plus_3_3xkernelwise_1D                       = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_2_3xkernelwise_1D                       = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_1_3xkernelwise_1D                       = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_0_3xkernelwise_1D                       = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);

subExperimentDir = 'compRand';
fc_plus_3_3xcompRand                            = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_2_3xcompRand                            = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_1_3xcompRand                            = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_0_3xcompRand                            = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);

subExperimentDir = 'layerwise-1D-from-CIFAR';
fc_plus_3_3xlayerwise_1D_from_CIFAR             = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_2_3xlayerwise_1D_from_CIFAR             = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_1_3xlayerwise_1D_from_CIFAR             = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_0_3xlayerwise_1D_from_CIFAR             = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);

subExperimentDir = 'layerwise-1D-from-COIL-100';
fc_plus_3_3xlayerwise_1D_from_COIL_100          = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_2_3xlayerwise_1D_from_COIL_100          = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_1_3xlayerwise_1D_from_COIL_100          = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_0_3xlayerwise_1D_from_COIL_100          = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);

subExperimentDir = 'layerwise-1D-from-MNIST';
fc_plus_3_3xlayerwise_1D_from_MNIST             = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_2_3xlayerwise_1D_from_MNIST             = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_1_3xlayerwise_1D_from_MNIST             = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_0_3xlayerwise_1D_from_MNIST             = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);

subExperimentDir = 'layerwise-1D-from-STL-10';
fc_plus_3_3xlayerwise_1D_from_STL_10            = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_2_3xlayerwise_1D_from_STL_10            = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_1_3xlayerwise_1D_from_STL_10            = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_0_3xlayerwise_1D_from_STL_10            = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);

subExperimentDir = '4-clustered-layerwise-1D-from-CIFAR';
fc_plus_3_3x4_clustered_layerwise_1D_from_CIFAR = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_2_3x4_clustered_layerwise_1D_from_CIFAR = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_1_3x4_clustered_layerwise_1D_from_CIFAR = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_0_3x4_clustered_layerwise_1D_from_CIFAR = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);

subExperimentDir = '8-clustered-layerwise-1D-from-CIFAR';
fc_plus_3_3x8_clustered_layerwise_1D_from_CIFAR = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_2_3x8_clustered_layerwise_1D_from_CIFAR = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_1_3x8_clustered_layerwise_1D_from_CIFAR = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);
fc_plus_0_3x8_clustered_layerwise_1D_from_CIFAR = loadEpochFile(resultsDir, experimentDir, subExperimentDir, 13); i = printLoadSuccess(i);

fprintf('\nDone!\n\n');
switch networkArch
  case 'alexnet'
    backPropDepthLimit = 5;
  case 'lenet'
    backPropDepthLimit = 3;
  case 'mnistnet'
    backPropDepthLimit = 2;
end

startEpoch = 1;
granularity = 1;
for backPropDepth = 0:backPropDepthLimit
  for resultType = {'train', 'val'}
    resultType = char(resultType);
    h = figure;
    experiment = sprintf('Varying Weight Initialization - FC + %d', backPropDepth);

    exp_1 = eval(sprintf('fc_plus_%d_3xkernelwise_1D', backPropDepth));
    exp_2 = eval(sprintf('fc_plus_%d_3xcompRand', backPropDepth));
    exp_3 = eval(sprintf('fc_plus_%d_3xlayerwise_1D_from_CIFAR', backPropDepth));
    exp_4 = eval(sprintf('fc_plus_%d_3xlayerwise_1D_from_COIL_100', backPropDepth));
    exp_5 = eval(sprintf('fc_plus_%d_3xlayerwise_1D_from_MNIST', backPropDepth));
    exp_6 = eval(sprintf('fc_plus_%d_3xlayerwise_1D_from_STL_10', backPropDepth));
    exp_7 = eval(sprintf('fc_plus_%d_3x4_clustered_layerwise_1D_from_CIFAR', backPropDepth));
    exp_8 = eval(sprintf('fc_plus_%d_3x8_clustered_layerwise_1D_from_CIFAR', backPropDepth));

    plot( ...
      startEpoch:1:epochNum, [exp_1.info.(resultType).error(1,startEpoch:epochNum)], 'r', ...
      startEpoch:1:epochNum, [exp_2.info.(resultType).error(1,startEpoch:epochNum)], 'k', ...
      startEpoch:1:epochNum, [exp_3.info.(resultType).error(1,startEpoch:epochNum)], 'r--', ...
      startEpoch:1:epochNum, [exp_4.info.(resultType).error(1,startEpoch:epochNum)], 'r-.', ...
      startEpoch:1:epochNum, [exp_5.info.(resultType).error(1,startEpoch:epochNum)], 'r-^', ...
      startEpoch:1:epochNum, [exp_6.info.(resultType).error(1,startEpoch:epochNum)], 'r:', ...
      startEpoch:1:epochNum, [exp_7.info.(resultType).error(1,startEpoch:epochNum)], 'y--', ...
      startEpoch:1:epochNum, [exp_8.info.(resultType).error(1,startEpoch:epochNum)], 'y:', ...
      'LineWidth', 2);
    grid on
    title(experiment);
    legend(...
      '3 x kernelwise 1D', ...
      '3 x compRand', ...
      '3 x layerwise 1D from CIFAR', ...
      '3 x layerwise 1D from COIL-100', ...
      '3 x layerwise 1D from MNIST', ...
      '3 x layerwise 1D from STL-10', ...
      '3 x 4-clustered layerwise 1D from CIFAR', ...
      '3 x 8-clustered layerwise 1D from CIFAR');
    xlabel('epoch')
    % ylabel('Training Error');
    ylim([0, 1 / granularity]);
    switch resultType
      case 'train'
        ylabel('Training Error');
        fileName = sprintf('Training Comparison - %s.png', experiment);
      case 'val'
        ylabel('Validation Error');
        fileName = sprintf('Validation Comparison - %s.png', experiment);
    end
    saveas(h, fileName);

    if backPropDepth == 0 && strcmp(resultType, 'val')
      fprintf('exp 1 converges to %6.5f\n', 1 - exp_1.info.val.error(1,epochNum));
      fprintf('exp 2 converges to %6.5f\n', 1 - exp_2.info.val.error(1,epochNum));
      fprintf('exp 3 converges to %6.5f\n', 1 - exp_3.info.val.error(1,epochNum));
      fprintf('exp 4 converges to %6.5f\n', 1 - exp_4.info.val.error(1,epochNum));
      fprintf('exp 5 converges to %6.5f\n', 1 - exp_5.info.val.error(1,epochNum));
      fprintf('exp 6 converges to %6.5f\n', 1 - exp_6.info.val.error(1,epochNum));
      fprintf('exp 7 converges to %6.5f\n', 1 - exp_7.info.val.error(1,epochNum));
      fprintf('exp 8 converges to %6.5f\n', 1 - exp_8.info.val.error(1,epochNum));
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
function epochFile = loadEpochFile(resultsDir, experimentDir, subExperimentDir, bpd);
% --------------------------------------------------------------------
  epochFile = load(fullfile( ...
    resultsDir, ...
    experimentDir, ...
    subExperimentDir, ...
    findSubDirWithBackPropDepth(fullfile(resultsDir, experimentDir, subExperimentDir), bpd), ...
    epochFile));

% --------------------------------------------------------------------
function i = printLoadSuccess(i);
% --------------------------------------------------------------------
  fprintf('\t[INFO] file %d loaded successfully.\n', i);
  i = i + 1;
