dataDir = '/Volumes/Amir/results/';

epochNum = 50;
epochFile = sprintf('net-epoch-%d.mat', epochNum);
fprintf('Loading files...'); i = 1;

dataset = 'stl-10';
networkArch = 'lenet';

switch dataset
  case 'cifar'
    experimentDir = '2016-11-18-21; Varying Layerwise Weight Initialization; CIFAR; LeNet; FC+{0-3}';
    subDataDir = 'kernelwise-1D';
    fc_plus_3_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-18-Nov-2016-20-01-04-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-18-Nov-2016-20-34-20-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-18-Nov-2016-21-00-26-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-18-Nov-2016-21-20-55-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = 'compRand';
    fc_plus_3_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-18-Nov-2016-18-25-15-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-18-Nov-2016-18-56-58-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-18-Nov-2016-19-22-55-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-18-Nov-2016-19-44-24-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = 'layerwise-1D-from-CIFAR';
    fc_plus_3_3xlayerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-18-Nov-2016-19-52-57-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3xlayerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-18-Nov-2016-20-26-00-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3xlayerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-18-Nov-2016-20-53-17-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3xlayerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-18-Nov-2016-21-13-56-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = 'layerwise-1D-from-COIL-100';
    fc_plus_3_3xlayerwise_1D_from_COIL_100 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-19-Nov-2016-12-13-44-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3xlayerwise_1D_from_COIL_100 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-19-Nov-2016-12-45-45-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3xlayerwise_1D_from_COIL_100 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-19-Nov-2016-13-11-51-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3xlayerwise_1D_from_COIL_100 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-19-Nov-2016-13-33-02-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = 'layerwise-1D-from-MNIST';
    fc_plus_3_3xlayerwise_1D_from_MNIST = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-20-Nov-2016-23-51-13-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3xlayerwise_1D_from_MNIST = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-21-Nov-2016-00-23-49-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3xlayerwise_1D_from_MNIST = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-21-Nov-2016-00-50-02-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3xlayerwise_1D_from_MNIST = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-21-Nov-2016-01-10-18-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = 'layerwise-1D-from-STL-10';
    fc_plus_3_3xlayerwise_1D_from_STL_10 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-19-Nov-2016-13-49-42-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3xlayerwise_1D_from_STL_10 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-19-Nov-2016-14-21-07-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3xlayerwise_1D_from_STL_10 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-19-Nov-2016-14-47-26-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3xlayerwise_1D_from_STL_10 = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-19-Nov-2016-15-07-53-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = '4-clustered-layerwise-1D-from-CIFAR';
    fc_plus_3_3x4_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-21-Nov-2016-16-35-08-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3x4_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-21-Nov-2016-17-07-52-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3x4_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-21-Nov-2016-17-34-35-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3x4_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-21-Nov-2016-17-55-32-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = '8-clustered-layerwise-1D-from-CIFAR';
    fc_plus_3_3x8_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-21-Nov-2016-16-35-04-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3x8_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-21-Nov-2016-17-07-59-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3x8_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-21-Nov-2016-17-34-57-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3x8_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'cifar-lenet-21-Nov-2016-17-56-01-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
  case 'coil-100'
    experimentDir = '2016-11-18-21; Varying Layerwise Weight Initialization; COIL-100; LeNet; FC+{0-3}';
    subDataDir = 'kernelwise-1D';
    fc_plus_3_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-26-00-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-29-55-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-33-05-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-35-52-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = 'compRand';
    fc_plus_3_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-14-01-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-17-39-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-20-49-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-23-34-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = 'layerwise-1D-from-CIFAR';
    fc_plus_3_3xlayerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-19-Nov-2016-13-10-59-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3xlayerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-19-Nov-2016-13-15-03-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3xlayerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-19-Nov-2016-13-18-22-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3xlayerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-19-Nov-2016-13-21-18-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = 'layerwise-1D-from-COIL-100';
    fc_plus_3_3xlayerwise_1D_from_COIL_100 = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-38-21-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3xlayerwise_1D_from_COIL_100 = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-43-03-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3xlayerwise_1D_from_COIL_100 = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-46-19-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3xlayerwise_1D_from_COIL_100 = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-18-Nov-2016-19-49-25-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = 'layerwise-1D-from-MNIST';
    fc_plus_3_3xlayerwise_1D_from_MNIST = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-21-Nov-2016-01-46-03-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3xlayerwise_1D_from_MNIST = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-21-Nov-2016-01-49-54-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3xlayerwise_1D_from_MNIST = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-21-Nov-2016-01-53-11-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3xlayerwise_1D_from_MNIST = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-21-Nov-2016-01-55-57-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = 'layerwise-1D-from-STL-10';
    fc_plus_3_3xlayerwise_1D_from_STL_10 = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-19-Nov-2016-13-36-24-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3xlayerwise_1D_from_STL_10 = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-19-Nov-2016-13-40-08-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3xlayerwise_1D_from_STL_10 = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-19-Nov-2016-13-43-24-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3xlayerwise_1D_from_STL_10 = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-19-Nov-2016-13-46-17-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = '4-clustered-layerwise-1D-from-CIFAR';
    fc_plus_3_3x4_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-21-Nov-2016-18-12-10-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3x4_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-21-Nov-2016-18-15-54-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3x4_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-21-Nov-2016-18-19-09-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3x4_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-21-Nov-2016-18-21-58-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = '8-clustered-layerwise-1D-from-CIFAR';
    fc_plus_3_3x8_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-21-Nov-2016-18-12-44-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3x8_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-21-Nov-2016-18-16-30-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3x8_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-21-Nov-2016-18-19-47-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3x8_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'coil-100-lenet-21-Nov-2016-18-22-39-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
  case 'mnist'
    experimentDir = '2016-11-18-21; Varying Layerwise Weight Initialization; MNIST; LeNet; FC+{0-3}';
    subDataDir = 'kernelwise-1D';
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

    subDataDir = '4-clustered-layerwise-1D-from-CIFAR';
    fc_plus_3_3x4_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-18-24-29-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3x4_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-19-02-49-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3x4_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-19-33-43-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3x4_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-19-58-17-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = '8-clustered-layerwise-1D-from-CIFAR';
    fc_plus_3_3x8_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-18-25-12-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3x8_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-19-03-42-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3x8_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-19-34-58-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3x8_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'mnist-lenet-21-Nov-2016-19-59-47-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
  case 'stl-10'
    experimentDir = '2016-11-18-21; Varying Layerwise Weight Initialization; STL-10; LeNet; FC+{0-3}';
    subDataDir = 'kernelwise-1D';
    fc_plus_3_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-24-40-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-30-24-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-35-23-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3x1D = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-39-46-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = 'compRand';
    fc_plus_3_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-04-48-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-10-20-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-15-20-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3xcompRand = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-19-41-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = 'layerwise-1D-from-CIFAR';
    fc_plus_3_3xlayerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-19-Nov-2016-12-13-46-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3xlayerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-19-Nov-2016-12-19-26-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3xlayerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-19-Nov-2016-12-24-31-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3xlayerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-19-Nov-2016-12-28-58-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = 'layerwise-1D-from-COIL-100';
    fc_plus_3_3xlayerwise_1D_from_COIL_100 = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-19-Nov-2016-12-33-00-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3xlayerwise_1D_from_COIL_100 = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-19-Nov-2016-12-38-49-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3xlayerwise_1D_from_COIL_100 = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-19-Nov-2016-12-43-46-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3xlayerwise_1D_from_COIL_100 = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-19-Nov-2016-12-48-08-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = 'layerwise-1D-from-MNIST';
    fc_plus_3_3xlayerwise_1D_from_MNIST = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-21-Nov-2016-01-26-41-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3xlayerwise_1D_from_MNIST = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-21-Nov-2016-01-32-25-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3xlayerwise_1D_from_MNIST = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-21-Nov-2016-01-37-26-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3xlayerwise_1D_from_MNIST = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-21-Nov-2016-01-41-59-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = 'layerwise-1D-from-STL-10';
    fc_plus_3_3xlayerwise_1D_from_STL_10 = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-43-42-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3xlayerwise_1D_from_STL_10 = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-49-14-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3xlayerwise_1D_from_STL_10 = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-54-08-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3xlayerwise_1D_from_STL_10 = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-18-Nov-2016-18-58-30-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = '4-clustered-layerwise-1D-from-CIFAR';
    fc_plus_3_3x4_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-21-Nov-2016-20-17-30-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3x4_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-21-Nov-2016-20-23-03-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3x4_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-21-Nov-2016-20-27-53-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3x4_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-21-Nov-2016-20-32-12-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

    subDataDir = '8-clustered-layerwise-1D-from-CIFAR';
    fc_plus_3_3x8_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-21-Nov-2016-20-19-08-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_2_3x8_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-21-Nov-2016-20-24-44-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_1_3x8_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-21-Nov-2016-20-29-38-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
    fc_plus_0_3x8_clustered_layerwise_1D_from_CIFAR = load(fullfile(dataDir, experimentDir, subDataDir, 'stl-10-lenet-21-Nov-2016-20-34-00-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
end

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

    exp_1 = eval(sprintf('fc_plus_%d_3x1D', backPropDepth));
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
