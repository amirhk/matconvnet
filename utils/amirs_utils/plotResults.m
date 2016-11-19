dataDir = '/Volumes/Amir/results/';
experimentDir = '2016-11-18-18; Varying Layerwise Weight Initialization; COIL-100; LeNet; FC+{0-3}';

epochNum = 50;
epochFile = sprintf('net-epoch-%d.mat', epochNum);
fprintf('Loading files...'); i = 1;

dataset = 'stl-10';
networkArch = 'lenet';

subDataDir = 'compRand';
fc_plus_3_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-14-01-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-17-39-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-20-49-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-23-34-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '1D';
fc_plus_3_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-26-00-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-29-55-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-33-05-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-35-52-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = 'layerwise-1D';
fc_plus_3_3xlayerwise_1D = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-38-21-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3xlayerwise_1D = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-43-03-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3xlayerwise_1D = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-46-19-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3xlayerwise_1D = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-49-25-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

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
