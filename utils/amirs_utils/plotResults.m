dataDir = '/Volumes/Amir/results/';
experimentDir = '2016-11-18-18; Varying Layerwise Weight Initialization; CIFAR; LeNet; FC+{0-3}';

epochNum = 50;
epochFile = sprintf('net-epoch-%d.mat', epochNum);
fprintf('Loading files...'); i = 1;

dataset = 'stl-10';
networkArch = 'lenet';

subDataDir = 'compRand';
fc_plus_3_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-04-48-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-10-20-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-15-20-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-19-41-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '1D';
fc_plus_3_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-24-40-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-30-24-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-35-23-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-39-46-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = 'layerwise-1D';
fc_plus_3_3xlayerwise_1D = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-43-42-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3xlayerwise_1D = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-49-14-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3xlayerwise_1D = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-54-08-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3xlayerwise_1D = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-58-30-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

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
    experiment = sprintf('Varying Weight Initialization With Dropout - FC + %d', backPropDepth);

    exp_1 = eval(sprintf('fc_plus_%d_3xcompRand', backPropDepth));
    exp_2 = eval(sprintf('fc_plus_%d_3x1D', backPropDepth));
    exp_3 = eval(sprintf('fc_plus_%d_3xlayerwise_1D', backPropDepth));

    plot( ...
      startEpoch:1:epochNum, [exp_1.info.(resultType).error(1,startEpoch:epochNum)], 'k', ...
      startEpoch:1:epochNum, [exp_2.info.(resultType).error(1,startEpoch:epochNum)], 'r', ...
      startEpoch:1:epochNum, [exp_3.info.(resultType).error(1,startEpoch:epochNum)], 'r--', ...
      'LineWidth', 2);
    grid on
    title(experiment);
    legend(...
      '3 x compRand', ...
      '3 x 1D', ...
      '3 x layerwise 1D');
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
