dataDir = '/Volumes/Amir/results/';
experimentDir = '2016-11-18-21; Varying Layerwise Weight Initialization; MNIST; LeNet; FC+{0-3}';

epochNum = 50;
epochFile = sprintf('net-epoch-%d.mat', epochNum);
fprintf('Loading files...'); i = 1;

dataset = 'mnist';
networkArch = 'lenet';

subDataDir = '1D';
fc_plus_3_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-01-44-03-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-02-20-56-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-02-50-30-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-03-13-47-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = 'compRand';
fc_plus_3_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-20-Nov-2016-20-47-41-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-20-Nov-2016-21-25-29-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-20-Nov-2016-21-56-14-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-20-Nov-2016-22-20-13-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = 'layerwise-1D-from-CIFAR';
fc_plus_3_3xlayerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-03-32-27-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3xlayerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-04-08-38-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3xlayerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-04-38-11-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3xlayerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-05-01-23-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = 'layerwise-1D-from-COIL-100';
fc_plus_3_3xlayerwise_1D_from_COIL_100 = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-05-20-02-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3xlayerwise_1D_from_COIL_100 = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-05-56-13-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3xlayerwise_1D_from_COIL_100 = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-06-25-44-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3xlayerwise_1D_from_COIL_100 = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-06-49-02-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = 'layerwise-1D-from-MNIST';
fc_plus_3_3xlayerwise_1D_from_MNIST = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-07-07-45-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3xlayerwise_1D_from_MNIST = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-07-44-08-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3xlayerwise_1D_from_MNIST = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-08-13-50-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3xlayerwise_1D_from_MNIST = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-08-37-07-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = 'layerwise-1D-from-STL-10';
fc_plus_3_3xlayerwise_1D_from_STL_10 = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-08-55-41-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3xlayerwise_1D_from_STL_10 = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-09-31-50-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3xlayerwise_1D_from_STL_10 = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-10-02-08-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3xlayerwise_1D_from_STL_10 = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-10-26-04-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

fprintf('\nDone!');
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

    exp_1 = eval(sprintf('fc_plus_%d_3x1D', backPropDepth));
    exp_2 = eval(sprintf('fc_plus_%d_3xcompRand', backPropDepth));
    exp_3 = eval(sprintf('fc_plus_%d_3xlayerwise_1D_from_CIFAR', backPropDepth));
    exp_4 = eval(sprintf('fc_plus_%d_3xlayerwise_1D_from_COIL_100', backPropDepth));
    exp_5 = eval(sprintf('fc_plus_%d_3xlayerwise_1D_from_MNIST', backPropDepth));
    exp_6 = eval(sprintf('fc_plus_%d_3xlayerwise_1D_from_STL_10', backPropDepth));

    plot( ...
      startEpoch:1:epochNum, [exp_1.info.(resultType).error(1,startEpoch:epochNum)], 'r', ...
      startEpoch:1:epochNum, [exp_2.info.(resultType).error(1,startEpoch:epochNum)], 'k', ...
      startEpoch:1:epochNum, [exp_3.info.(resultType).error(1,startEpoch:epochNum)], 'r--', ...
      startEpoch:1:epochNum, [exp_4.info.(resultType).error(1,startEpoch:epochNum)], 'r-.', ...
      startEpoch:1:epochNum, [exp_5.info.(resultType).error(1,startEpoch:epochNum)], 'r-^', ...
      startEpoch:1:epochNum, [exp_6.info.(resultType).error(1,startEpoch:epochNum)], 'r:', ...
      'LineWidth', 2);
    grid on
    title(experiment);
    legend(...
      '3 x 1D', ...
      '3 x compRand', ...
      '3 x layerwise 1D from CIFAR', ...
      '3 x layerwise 1D from COIL-100', ...
      '3 x layerwise 1D from MNIST (default)', ...
      '3 x layerwise 1D from STL-10');
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
  end
end
