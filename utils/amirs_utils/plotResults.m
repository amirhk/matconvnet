dataDir = '/Volumes/Amir/results/';

epochNum = 50;
epochFile = sprintf('net-epoch-%d.mat', epochNum);
fprintf('Loading files...'); i = 1;

subDataDir = '2016-11-13-14; CIFAR; LeNet; FC+{0-3}; 1x2D_mult_randn_2x1D';
fc_plus_3_1x2D_mult_randn_2x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-11-08-29-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_1x2D_mult_randn_2x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-11-41-24-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_1x2D_mult_randn_2x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-12-08-31-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_1x2D_mult_randn_2x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-12-30-02-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-13-14; CIFAR; LeNet; FC+{0-3}; 1x2D_mult_randn_2xcompRand';
fc_plus_3_1x2D_mult_randn_2xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-12-47-03-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_1x2D_mult_randn_2xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-13-19-59-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_1x2D_mult_randn_2xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-13-46-54-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_1x2D_mult_randn_2xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-14-08-10-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-13-14; CIFAR; LeNet; FC+{0-3}; 1x2D_shiftflip_2x1D';
fc_plus_3_1x2D_shiftflip_2x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-11-08-30-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_1x2D_shiftflip_2x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-11-42-36-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_1x2D_shiftflip_2x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-12-10-14-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_1x2D_shiftflip_2x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-12-32-03-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-13-14; CIFAR; LeNet; FC+{0-3}; 1x2D_shiftflip_2xcompRand';
fc_plus_3_1x2D_shiftflip_2xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-12-49-16-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_1x2D_shiftflip_2xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-13-23-04-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_1x2D_shiftflip_2xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-13-50-37-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_1x2D_shiftflip_2xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-14-12-17-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-13-14; CIFAR; LeNet; FC+{0-3}; 2x2D_mult_randn_1x1D';
fc_plus_3_2x2D_mult_randn_1x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-07-28-36-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_2x2D_mult_randn_1x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-08-03-52-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_2x2D_mult_randn_1x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-08-33-19-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_2x2D_mult_randn_1x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-08-56-39-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-13-14; CIFAR; LeNet; FC+{0-3}; 2x2D_mult_randn_1xcompRand';
fc_plus_3_2x2D_mult_randn_1xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-09-14-18-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_2x2D_mult_randn_1xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-09-48-47-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_2x2D_mult_randn_1xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-10-15-39-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_2x2D_mult_randn_1xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-10-37-00-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-13-14; CIFAR; LeNet; FC+{0-3}; 2x2D_shiftflip_1x1D';
fc_plus_3_2x2D_shiftflip_1x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-03-58-31-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_2x2D_shiftflip_1x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-04-33-18-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_2x2D_shiftflip_1x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-05-02-28-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_2x2D_shiftflip_1x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-05-25-37-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-13-14; CIFAR; LeNet; FC+{0-3}; 2x2D_shiftflip_1xcompRand';
fc_plus_3_2x2D_shiftflip_1xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-05-43-21-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_2x2D_shiftflip_1xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-06-18-14-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_2x2D_shiftflip_1xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-06-47-30-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_2x2D_shiftflip_1xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-07-10-40-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-13-14; CIFAR; LeNet; FC+{0-3}; 3x1D';
fc_plus_3_3x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-02-13-29-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-02-48-46-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-03-17-46-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-03-40-53-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-13-14; CIFAR; LeNet; FC+{0-3}; 3xbaseline';
fc_plus_3_3xbaseline = load(fullfile(dataDir, subDataDir, 'cifar-lenet-13-Nov-2016-22-52-12-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3xbaseline = load(fullfile(dataDir, subDataDir, 'cifar-lenet-13-Nov-2016-23-25-24-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3xbaseline = load(fullfile(dataDir, subDataDir, 'cifar-lenet-13-Nov-2016-23-52-34-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3xbaseline = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-00-14-08-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-13-14; CIFAR; LeNet; FC+{0-3}; 3xcompRand';
fc_plus_3_3xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-00-31-14-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-01-04-13-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-01-32-30-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-01-55-38-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

fprintf('\nDone!');

for backPropDepth = 0:3
  for resultType = {'train', 'val'}
    resultType = char(resultType);
    h = figure;
    experiment = sprintf('Varying Weight Initialization Schemes - FC + %d', backPropDepth);
    exp_1 = eval(sprintf('fc_plus_%d_2x2D_mult_randn_1x1D', backPropDepth));
    exp_2 = eval(sprintf('fc_plus_%d_2x2D_mult_randn_1xcompRand', backPropDepth));
    exp_3 = eval(sprintf('fc_plus_%d_2x2D_shiftflip_1x1D', backPropDepth));
    exp_4 = eval(sprintf('fc_plus_%d_2x2D_shiftflip_1xcompRand', backPropDepth));
    exp_5 = eval(sprintf('fc_plus_%d_3x1D', backPropDepth));
    % exp_6 = eval(sprintf('fc_plus_%d_3xbaseline', backPropDepth));
    exp_7 = eval(sprintf('fc_plus_%d_3xcompRand', backPropDepth));
    exp_8 = eval(sprintf('fc_plus_%d_1x2D_mult_randn_2x1D', backPropDepth));
    exp_9 = eval(sprintf('fc_plus_%d_1x2D_mult_randn_2xcompRand', backPropDepth));
    exp_10 = eval(sprintf('fc_plus_%d_1x2D_shiftflip_2x1D', backPropDepth));
    exp_11 = eval(sprintf('fc_plus_%d_1x2D_shiftflip_2xcompRand', backPropDepth));

    plot( ...
      1:1:epochNum, [exp_1.info.(resultType).error(1,1:epochNum)], 'y', ...
      1:1:epochNum, [exp_2.info.(resultType).error(1,1:epochNum)], 'y--', ...
      1:1:epochNum, [exp_3.info.(resultType).error(1,1:epochNum)], 'm', ...
      1:1:epochNum, [exp_4.info.(resultType).error(1,1:epochNum)], 'm--', ...
      1:1:epochNum, [exp_5.info.(resultType).error(1,1:epochNum)], 'r', ...
      1:1:epochNum, [exp_7.info.(resultType).error(1,1:epochNum)], 'k', ...
      1:1:epochNum, [exp_8.info.(resultType).error(1,1:epochNum)], 'g', ...
      1:1:epochNum, [exp_9.info.(resultType).error(1,1:epochNum)], 'g--', ...
      1:1:epochNum, [exp_10.info.(resultType).error(1,1:epochNum)], 'b', ...
      1:1:epochNum, [exp_11.info.(resultType).error(1,1:epochNum)], 'b--', ...
      'LineWidth', 2);
    % 1:1:epochNum, [exp_6.info.(resultType).error(1,1:epochNum)], 'c', ...
    grid on
    title(experiment);
    legend(...
      '2x2D mult randn + 1x1D', ...
      '2x2D mult randn + 1xcompRand', ...
      '2x2D shiftflip + 1x1D', ...
      '2x2D shiftflip + 1xcompRand', ...
      '3x1D', ...
      '3xcompRand', ...
      '1x2D mult randn + 2x1D', ...
      '1x2D mult randn + 2xcompRand', ...
      '1x2D shiftflip + 2x1D', ...
      '1x2D shiftflip + 2xcompRand');
    % '3xbaseline (pre-train)', ...
    xlabel('epoch')
    % ylabel('Training Error');
    ylim([0,1]);
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
