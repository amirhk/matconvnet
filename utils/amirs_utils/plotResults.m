dataDir = '/Volumes/Amir/results/';

epochNum = 50;
epochFile = sprintf('net-epoch-%d.mat', epochNum);
fprintf('Loading files...'); i = 1;

subDataDir = '2016-11-10-14; STL10; LeNet; FC+{0-3}; 1x2D_mult_randn_2x1D';
fc_plus_3_1x2D_mult_randn_2x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-10-42-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_1x2D_mult_randn_2x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-16-32-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_1x2D_mult_randn_2x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-21-44-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_1x2D_mult_randn_2x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-26-22-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-10-14; STL10; LeNet; FC+{0-3}; 1x2D_mult_randn_2xcompRand';
fc_plus_3_1x2D_mult_randn_2xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-30-31-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_1x2D_mult_randn_2xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-36-23-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_1x2D_mult_randn_2xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-41-36-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_1x2D_mult_randn_2xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-46-15-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-10-14; STL10; LeNet; FC+{0-3}; 1x2D_shiftflip_2x1D';
fc_plus_3_1x2D_shiftflip_2x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-14-30-02-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_1x2D_shiftflip_2x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-14-35-58-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_1x2D_shiftflip_2x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-14-41-12-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_1x2D_shiftflip_2x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-14-46-04-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-10-14; STL10; LeNet; FC+{0-3}; 1x2D_shiftflip_2xcompRand';
fc_plus_3_1x2D_shiftflip_2xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-14-50-15-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_1x2D_shiftflip_2xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-14-56-04-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_1x2D_shiftflip_2xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-01-21-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_1x2D_shiftflip_2xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-06-16-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-10-14; STL10; LeNet; FC+{0-3}; 2x2D_mult_randn_1x1D';
fc_plus_3_2x2D_mult_randn_1x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-29-55-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_2x2D_mult_randn_1x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-35-42-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_2x2D_mult_randn_1x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-40-52-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_2x2D_mult_randn_1x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-45-30-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-10-14; STL10; LeNet; FC+{0-3}; 2x2D_mult_randn_1xcompRand';
fc_plus_3_2x2D_mult_randn_1xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-49-41-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_2x2D_mult_randn_1xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-55-32-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_2x2D_mult_randn_1xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-16-00-47-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_2x2D_mult_randn_1xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-16-05-26-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-10-14; STL10; LeNet; FC+{0-3}; 2x2D_shiftflip_1x1D';
fc_plus_3_2x2D_shiftflip_1x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-50-25-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_2x2D_shiftflip_1x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-56-18-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_2x2D_shiftflip_1x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-16-01-34-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_2x2D_shiftflip_1x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-16-06-13-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-10-14; STL10; LeNet; FC+{0-3}; 2x2D_shiftflip_1xcompRand';
fc_plus_3_2x2D_shiftflip_1xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-16-10-27-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_2x2D_shiftflip_1xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-16-16-35-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_2x2D_shiftflip_1xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-16-21-55-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_2x2D_shiftflip_1xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-16-26-37-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-10-14; STL10; LeNet; FC+{0-3}; 3x1D';
fc_plus_3_3x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-10-17-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-16-03-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-21-12-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-25-47-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-10-14; STL10; LeNet; FC+{0-3}; 3xbaseline';
fc_plus_3_3xbaseline = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-14-30-00-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3xbaseline = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-14-35-48-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3xbaseline = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-14-40-58-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3xbaseline = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-14-45-53-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

subDataDir = '2016-11-10-14; STL10; LeNet; FC+{0-3}; 3xcompRand';
fc_plus_3_3xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-14-50-00-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_2_3xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-14-55-44-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_1_3xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-00-54-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
fc_plus_0_3xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-05-48-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

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
