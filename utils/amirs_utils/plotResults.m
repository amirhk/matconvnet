dataDir = '/Volumes/Amir/results/';
experimentDir = '2016-11-17-17; Varying Weight Initialization with Dropout; CIFAR; LeNet; FC+{0-3}';

epochNum = 50;
epochFile = sprintf('net-epoch-%d.mat', epochNum);
fprintf('Loading files...'); i = 1;

dataset = 'coil-100';
networkArch = 'lenet';


subDataDir = '2016-11-17-17; CIFAR; LeNet; FC+{0-3}; 3x1D; 0xdropout';
fc_plus_3_3x1D_0xdropout = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-21-03-07-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3x1D_0xdropout = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-21-35-28-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3x1D_0xdropout = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-22-02-13-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3x1D_0xdropout = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-22-23-22-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-17-17; CIFAR; LeNet; FC+{0-3}; 3x1D; 1xdropout after layer 1';
fc_plus_3_3x1D_1xdropout_after_1 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-21-05-08-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3x1D_1xdropout_after_1 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-21-38-05-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3x1D_1xdropout_after_1 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-22-05-17-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3x1D_1xdropout_after_1 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-22-26-48-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-17-17; CIFAR; LeNet; FC+{0-3}; 3x1D; 1xdropout after layer 3';
fc_plus_3_3x1D_1xdropout_after_3 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-14-11-32-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3x1D_1xdropout_after_3 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-14-45-07-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3x1D_1xdropout_after_3 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-15-12-03-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3x1D_1xdropout_after_3 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-15-33-20-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-17-17; CIFAR; LeNet; FC+{0-3}; 3x1D; 1xdropout in FC';
fc_plus_3_3x1D_1xdropout_in_fc = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-17-30-38-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3x1D_1xdropout_in_fc = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-18-03-26-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3x1D_1xdropout_in_fc = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-18-30-27-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3x1D_1xdropout_in_fc = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-18-51-57-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-17-17; CIFAR; LeNet; FC+{0-3}; 3x1D; 2xdropout after layers 1,3';
fc_plus_3_3x1D_2xdropout_after_1_3 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-17-31-20-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3x1D_2xdropout_after_1_3 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-18-04-24-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3x1D_2xdropout_after_1_3 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-18-32-01-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3x1D_2xdropout_after_1_3 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-18-53-43-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-17-17; CIFAR; LeNet; FC+{0-3}; 3xcompRand; 0xdropout';
fc_plus_3_3xcompRand_0xdropout = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-19-26-21-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3xcompRand_0xdropout = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-19-58-57-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3xcompRand_0xdropout = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-20-25-40-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3xcompRand_0xdropout = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-20-46-31-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-17-17; CIFAR; LeNet; FC+{0-3}; 3xcompRand; 1xdropout after layer 1';
fc_plus_3_3xcompRand_1xdropout_after_1 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-19-26-28-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3xcompRand_1xdropout_after_1 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-19-59-30-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3xcompRand_1xdropout_after_1 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-20-26-43-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3xcompRand_1xdropout_after_1 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-20-48-13-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-17-17; CIFAR; LeNet; FC+{0-3}; 3xcompRand; 1xdropout after layer 3';
fc_plus_3_3xcompRand_1xdropout_after_3 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-exp1-13-30-10-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3xcompRand_1xdropout_after_3 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-exp2-13-02-46-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3xcompRand_1xdropout_after_3 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-exp3-12-41-35-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3xcompRand_1xdropout_after_3 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-exp4-12-24-40-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-17-17; CIFAR; LeNet; FC+{0-3}; 3xcompRand; 1xdropout in FC';
fc_plus_3_3xcompRand_1xdropout_in_fc = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-15-51-40-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3xcompRand_1xdropout_in_fc = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-16-25-01-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3xcompRand_1xdropout_in_fc = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-16-51-53-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3xcompRand_1xdropout_in_fc = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-17-13-36-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-17-17; CIFAR; LeNet; FC+{0-3}; 3xcompRand; 2xdropout after layers 1,3';
fc_plus_3_3xcompRand_2xdropout_after_1_3 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-15-51-35-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3xcompRand_2xdropout_after_1_3 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-16-25-01-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3xcompRand_2xdropout_after_1_3 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-16-52-20-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3xcompRand_2xdropout_after_1_3 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-17-Nov-2016-17-14-08-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

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

    exp_1 = eval(sprintf('fc_plus_%d_3x1D_0xdropout', backPropDepth));
    exp_2 = eval(sprintf('fc_plus_%d_3x1D_1xdropout_after_1', backPropDepth));
    exp_3 = eval(sprintf('fc_plus_%d_3x1D_1xdropout_after_3', backPropDepth));
    exp_4 = eval(sprintf('fc_plus_%d_3x1D_1xdropout_in_fc', backPropDepth));
    exp_5 = eval(sprintf('fc_plus_%d_3x1D_2xdropout_after_1_3', backPropDepth));
    exp_6 = eval(sprintf('fc_plus_%d_3xcompRand_0xdropout', backPropDepth));
    exp_7 = eval(sprintf('fc_plus_%d_3xcompRand_1xdropout_after_1', backPropDepth));
    exp_8 = eval(sprintf('fc_plus_%d_3xcompRand_1xdropout_after_3', backPropDepth));
    exp_9 = eval(sprintf('fc_plus_%d_3xcompRand_1xdropout_in_fc', backPropDepth));
    exp_10 = eval(sprintf('fc_plus_%d_3xcompRand_2xdropout_after_1_3', backPropDepth));

    plot( ...
      startEpoch:1:epochNum, [exp_1.info.(resultType).error(1,startEpoch:epochNum)], 'r', ...
      startEpoch:1:epochNum, [exp_2.info.(resultType).error(1,startEpoch:epochNum)], 'r--', ...
      startEpoch:1:epochNum, [exp_3.info.(resultType).error(1,startEpoch:epochNum)], 'r-.', ...
      startEpoch:1:epochNum, [exp_4.info.(resultType).error(1,startEpoch:epochNum)], 'r-^', ...
      startEpoch:1:epochNum, [exp_5.info.(resultType).error(1,startEpoch:epochNum)], 'r:', ...
      startEpoch:1:epochNum, [exp_6.info.(resultType).error(1,startEpoch:epochNum)], 'k', ...
      startEpoch:1:epochNum, [exp_7.info.(resultType).error(1,startEpoch:epochNum)], 'k--', ...
      startEpoch:1:epochNum, [exp_8.info.(resultType).error(1,startEpoch:epochNum)], 'k-.', ...
      startEpoch:1:epochNum, [exp_9.info.(resultType).error(1,startEpoch:epochNum)], 'k-^', ...
      startEpoch:1:epochNum, [exp_10.info.(resultType).error(1,startEpoch:epochNum)], 'k:', ...
      'LineWidth', 2);
    grid on
    title(experiment);
    legend(...
      '3x1D 0xdropout', ...
      '3x1D 1xdropout after conv 1', ...
      '3x1D 1xdropout after conv 3', ...
      '3x1D 1xdropout in FC', ...
      '3x1D 2xdropout after conv 1,3', ...
      '3xcompRand 0xdropout', ...
      '3xcompRand 1xdropout after conv 1', ...
      '3xcompRand 1xdropout after conv 3', ...
      '3xcompRand 1xdropout in FC', ...
      '3xcompRand 2xdropout after conv 1,3');
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
