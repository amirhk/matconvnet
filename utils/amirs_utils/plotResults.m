dataDir = '/Volumes/Amir/results/';

epochNum = 50;
epochFile = sprintf('net-epoch-%d.mat', epochNum);
fprintf('Loading files...'); i = 1;

dataset = 'coil-100';
networkArch = 'lenet';
if strcmp(dataset, 'cifar')
  if strcmp(networkArch, 'lenet')
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % --       --       --       --       --       --       --       --       --       --       --       --       --       --       --       --       --       --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
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

    subDataDir = '2016-11-13-14; CIFAR; LeNet; FC+{0-3}; 3x1D';
    fc_plus_3_3x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-02-13-29-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-02-48-46-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-03-17-46-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3x1D = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-03-40-53-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = '2016-11-13-14; CIFAR; LeNet; FC+{0-3}; 3xcompRand';
    fc_plus_3_3xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-00-31-14-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-01-04-13-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-01-32-30-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-lenet-14-Nov-2016-01-55-38-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  elseif strcmp(networkArch, 'alexnet')
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % --       --       --       --       --       --       --       --       --       --       --       --       --       --       --       --       --       --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    subDataDir = '2016-11-04-06; CIFAR; AlexNet; FC+{0-5}; 2x2D_mult_randn_3x1D';
    fc_plus_5_1x2D_mult_randn_2x1D = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-7-Nov-2016-18-23-00-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_4_1x2D_mult_randn_2x1D = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-7-Nov-2016-20-44-55-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_3_1x2D_mult_randn_2x1D = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-7-Nov-2016-22-55-36-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_1x2D_mult_randn_2x1D = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-8-Nov-2016-00-29-53-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_1x2D_mult_randn_2x1D = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-8-Nov-2016-01-51-51-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_1x2D_mult_randn_2x1D = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-8-Nov-2016-03-05-07-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = '2016-11-04-06; CIFAR; AlexNet; FC+{0-5}; 2x2D_mult_randn_3xcompRand';
    fc_plus_5_1x2D_mult_randn_2xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-7-Nov-2016-16-00-02-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_4_1x2D_mult_randn_2xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-7-Nov-2016-18-12-29-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_3_1x2D_mult_randn_2xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-7-Nov-2016-20-13-59-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_1x2D_mult_randn_2xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-7-Nov-2016-21-40-05-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_1x2D_mult_randn_2xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-7-Nov-2016-22-53-50-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_1x2D_mult_randn_2xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-7-Nov-2016-23-59-29-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = '2016-11-04-06; CIFAR; AlexNet; FC+{0-5}; 2x2D_shiftflip_3x1D';
    fc_plus_5_1x2D_shiftflip_2x1D = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-8-Nov-2016-08-58-00-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_4_1x2D_shiftflip_2x1D = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-8-Nov-2016-11-18-16-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_3_1x2D_shiftflip_2x1D = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-8-Nov-2016-13-28-00-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_1x2D_shiftflip_2x1D = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-8-Nov-2016-15-01-08-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_1x2D_shiftflip_2x1D = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-8-Nov-2016-16-21-30-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_1x2D_shiftflip_2x1D = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-8-Nov-2016-17-33-03-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = '2016-11-04-06; CIFAR; AlexNet; FC+{0-5}; 2x2D_shiftflip_3xcompRand';
    fc_plus_5_1x2D_shiftflip_2xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-8-Nov-2016-08-58-02-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_4_1x2D_shiftflip_2xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-8-Nov-2016-11-08-25-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_3_1x2D_shiftflip_2xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-8-Nov-2016-13-12-15-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_1x2D_shiftflip_2xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-8-Nov-2016-14-39-21-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_1x2D_shiftflip_2xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-8-Nov-2016-15-54-09-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_1x2D_shiftflip_2xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-8-Nov-2016-17-00-36-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = '2016-11-06-07; CIFAR; AlexNet; FC+{0-5}; 5x1D';
    fc_plus_5_3x1D = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-6-Nov-2016-18-34-49-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_4_3x1D = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-6-Nov-2016-22-00-38-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_3_3x1D = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-7-Nov-2016-00-09-52-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3x1D = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-7-Nov-2016-01-42-54-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3x1D = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-7-Nov-2016-03-03-06-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3x1D = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-7-Nov-2016-04-14-34-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = '2016-11-06-07; CIFAR; AlexNet; FC+{0-5}; 5xcompRand';
    fc_plus_5_3xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-8-Nov-2016-18-03-44-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_4_3xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-8-Nov-2016-20-15-13-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_3_3xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-8-Nov-2016-22-18-03-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-8-Nov-2016-23-44-28-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-9-Nov-2016-00-58-39-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3xcompRand = load(fullfile(dataDir, subDataDir, 'cifar-alexnet-9-Nov-2016-02-04-06-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  end
elseif strcmp(dataset, 'stl-10')
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % --       --       --       --       --       --       --       --       --       --       --       --       --       --       --       --       --       --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
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

  subDataDir = '2016-11-10-14; STL10; LeNet; FC+{0-3}; 3x1D';
  fc_plus_3_3x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-10-17-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_2_3x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-16-03-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_1_3x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-21-12-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_0_3x1D = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-25-47-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

  subDataDir = '2016-11-10-14; STL10; LeNet; FC+{0-3}; 3xcompRand';
  fc_plus_3_3xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-14-50-00-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_2_3xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-14-55-44-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_1_3xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-00-54-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_0_3xcompRand = load(fullfile(dataDir, subDataDir, 'stl-10-lenet-14-Nov-2016-15-05-48-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
elseif strcmp(dataset, 'coil-100')
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % --       --       --       --       --       --       --       --       --       --       --       --       --       --       --       --       --       --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  subDataDir = '2016-11-16-16; COIL-100; LeNet; FC+{0-3}; 1x2D_mult_randn_2x1D';
  fc_plus_3_1x2D_mult_randn_2x1D = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-16-13-59-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_2_1x2D_mult_randn_2x1D = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-16-17-23-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_1_1x2D_mult_randn_2x1D = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-16-20-26-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_0_1x2D_mult_randn_2x1D = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-16-23-06-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

  subDataDir = '2016-11-16-16; COIL-100; LeNet; FC+{0-3}; 1x2D_mult_randn_2xcompRand';
  fc_plus_3_1x2D_mult_randn_2xcompRand = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-16-25-29-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_2_1x2D_mult_randn_2xcompRand = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-16-28-55-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_1_1x2D_mult_randn_2xcompRand = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-16-31-57-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_0_1x2D_mult_randn_2xcompRand = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-16-34-38-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

  subDataDir = '2016-11-16-16; COIL-100; LeNet; FC+{0-3}; 1x2D_shiftflip_2x1D';
  fc_plus_3_1x2D_shiftflip_2x1D = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-15-50-16-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_2_1x2D_shiftflip_2x1D = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-15-53-50-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_1_1x2D_shiftflip_2x1D = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-15-56-59-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_0_1x2D_shiftflip_2x1D = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-15-59-45-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

  subDataDir = '2016-11-16-16; COIL-100; LeNet; FC+{0-3}; 1x2D_shiftflip_2xcompRand';
  fc_plus_3_1x2D_shiftflip_2xcompRand = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-16-02-09-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_2_1x2D_shiftflip_2xcompRand = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-16-05-53-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_1_1x2D_shiftflip_2xcompRand = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-16-08-55-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_0_1x2D_shiftflip_2xcompRand = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-16-11-35-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

  subDataDir = '2016-11-16-16; COIL-100; LeNet; FC+{0-3}; 3x1D';
  fc_plus_3_3x1D = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-15-38-27-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_2_3x1D = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-15-41-54-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_1_3x1D = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-15-45-02-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_0_3x1D = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-15-47-47-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

  subDataDir = '2016-11-16-16; COIL-100; LeNet; FC+{0-3}; 3xcompRand';
  fc_plus_3_3xcompRand = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-15-18-18-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_2_3xcompRand = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-15-21-56-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_1_3xcompRand = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-15-25-06-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_0_3xcompRand = load(fullfile(dataDir, subDataDir, 'coil-100-lenet-16-Nov-2016-15-27-53-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
elseif strcmp(dataset, 'mnist')
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % --       --       --       --       --       --       --       --       --       --       --       --       --       --       --       --       --       --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  subDataDir = '2016-11-15-15; MNIST; MNISTNet; FC+{0-2}; 1x2D_mult_randn_2x1D';
  fc_plus_2_1x2D_mult_randn_2x1D = load(fullfile(dataDir, subDataDir, 'mnist-mnistnet-15-Nov-2016-13-58-56-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_1_1x2D_mult_randn_2x1D = load(fullfile(dataDir, subDataDir, 'mnist-mnistnet-15-Nov-2016-14-16-31-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_0_1x2D_mult_randn_2x1D = load(fullfile(dataDir, subDataDir, 'mnist-mnistnet-15-Nov-2016-14-29-29-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

  subDataDir = '2016-11-15-15; MNIST; MNISTNet; FC+{0-2}; 1x2D_mult_randn_2xcompRand';
  fc_plus_2_1x2D_mult_randn_2xcompRand = load(fullfile(dataDir, subDataDir, 'mnist-mnistnet-15-Nov-2016-14-39-02-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_1_1x2D_mult_randn_2xcompRand = load(fullfile(dataDir, subDataDir, 'mnist-mnistnet-15-Nov-2016-14-56-23-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_0_1x2D_mult_randn_2xcompRand = load(fullfile(dataDir, subDataDir, 'mnist-mnistnet-15-Nov-2016-15-09-37-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

  subDataDir = '2016-11-15-15; MNIST; MNISTNet; FC+{0-2}; 1x2D_shiftflip_2x1D';
  fc_plus_2_1x2D_shiftflip_2x1D = load(fullfile(dataDir, subDataDir, 'mnist-mnistnet-15-Nov-2016-12-38-20-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_1_1x2D_shiftflip_2x1D = load(fullfile(dataDir, subDataDir, 'mnist-mnistnet-15-Nov-2016-12-55-19-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_0_1x2D_shiftflip_2x1D = load(fullfile(dataDir, subDataDir, 'mnist-mnistnet-15-Nov-2016-13-08-26-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

  subDataDir = '2016-11-15-15; MNIST; MNISTNet; FC+{0-2}; 1x2D_shiftflip_2xcompRand';
  fc_plus_2_1x2D_shiftflip_2xcompRand = load(fullfile(dataDir, subDataDir, 'mnist-mnistnet-15-Nov-2016-13-18-10-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_1_1x2D_shiftflip_2xcompRand = load(fullfile(dataDir, subDataDir, 'mnist-mnistnet-15-Nov-2016-13-35-42-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_0_1x2D_shiftflip_2xcompRand = load(fullfile(dataDir, subDataDir, 'mnist-mnistnet-15-Nov-2016-13-49-06-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

  subDataDir = '2016-11-15-15; MNIST; MNISTNet; FC+{0-2}; 3x1D';
  fc_plus_2_3x1D = load(fullfile(dataDir, subDataDir, 'mnist-mnistnet-15-Nov-2016-11-59-10-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_1_3x1D = load(fullfile(dataDir, subDataDir, 'mnist-mnistnet-15-Nov-2016-12-16-11-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_0_3x1D = load(fullfile(dataDir, subDataDir, 'mnist-mnistnet-15-Nov-2016-12-28-57-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

  subDataDir = '2016-11-15-15; MNIST; MNISTNet; FC+{0-2}; 3xcompRand';
  fc_plus_2_3xcompRand = load(fullfile(dataDir, subDataDir, 'mnist-mnistnet-15-Nov-2016-09-23-01-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_1_3xcompRand = load(fullfile(dataDir, subDataDir, 'mnist-mnistnet-15-Nov-2016-09-41-00-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
  fc_plus_0_3xcompRand = load(fullfile(dataDir, subDataDir, 'mnist-mnistnet-15-Nov-2016-09-53-55-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
end

fprintf('\nDone!');
switch networkArch
  case 'alexnet'
    backPropDepthLimit = 5;
  case 'lenet'
    backPropDepthLimit = 3;
  case 'mnistnet'
    backPropDepthLimit = 2;
end

% startEpoch = 1;
startEpoch = 10;
for backPropDepth = 0:backPropDepthLimit
  for resultType = {'train', 'val'}
    resultType = char(resultType);
    h = figure;
    experiment = sprintf('Varying Weight Initialization Schemes - FC + %d', backPropDepth);
    exp_1 = eval(sprintf('fc_plus_%d_1x2D_mult_randn_2x1D', backPropDepth));
    exp_2 = eval(sprintf('fc_plus_%d_1x2D_mult_randn_2xcompRand', backPropDepth));
    exp_3 = eval(sprintf('fc_plus_%d_1x2D_shiftflip_2x1D', backPropDepth));
    exp_4 = eval(sprintf('fc_plus_%d_1x2D_shiftflip_2xcompRand', backPropDepth));
    exp_5 = eval(sprintf('fc_plus_%d_3x1D', backPropDepth));
    exp_6 = eval(sprintf('fc_plus_%d_3xcompRand', backPropDepth));

    plot( ...
      startEpoch:1:epochNum, [exp_1.info.(resultType).error(1,startEpoch:epochNum)], 'y', ...
      startEpoch:1:epochNum, [exp_2.info.(resultType).error(1,startEpoch:epochNum)], 'y--', ...
      startEpoch:1:epochNum, [exp_3.info.(resultType).error(1,startEpoch:epochNum)], 'm', ...
      startEpoch:1:epochNum, [exp_4.info.(resultType).error(1,startEpoch:epochNum)], 'm--', ...
      startEpoch:1:epochNum, [exp_5.info.(resultType).error(1,startEpoch:epochNum)], 'r', ...
      startEpoch:1:epochNum, [exp_6.info.(resultType).error(1,startEpoch:epochNum)], 'k', ...
      'LineWidth', 2);
    grid on
    title(experiment);
    legend(...
      '1x2D mult randn + 2x1D', ...
      '1x2D mult randn + 2xcompRand', ...
      '1x2D shiftflip + 2x1D', ...
      '1x2D shiftflip + 2xcompRand', ...
      '3x1D', ...
      '3xcompRand');
    xlabel('epoch')
    % ylabel('Training Error');
    % ylim([0,1]);
    ylim([0,.1]);
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
