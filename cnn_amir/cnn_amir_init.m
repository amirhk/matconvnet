function net = cnn_amir_init(varargin)
opts.networkType = 'test';
opts.dataset = 'cifar';
opts = vl_argparse(opts, varargin);

tic;
rng(0);
net.layers = {};
% Meta parameters
switch opts.networkType
  case 'test'
    net.meta.trainOpts.learningRate = [0.001 * ones(1,5) 0.01 * ones(1,10)];
  case 'alex-net'
    % weights random from pre-train 1D (with or without whitening)
    net.meta.trainOpts.learningRate = [0.01*ones(1,5)  0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,15) 0.00005*ones(1,15)];
    % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    % TESTING.... weights random from pre-train 2D-super (with whitening)
    % net.meta.trainOpts.learningRate = [0.01*ones(1,5)  0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,15) 0.00005*ones(1,15)];
    % net.meta.trainOpts.learningRate = [0.001*ones(1,10)];
    % net.meta.trainOpts.learningRate = [0.0005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,5) 0.0001*ones(1,15) 0.00005*ones(1,15)];
    % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    % TESTING weights completely random (goes down after 50 * 0.001 to %86 then after 230 epochs to ~%60)
    % net.meta.trainOpts.learningRate = [0.001*ones(1,100) 0.0005*ones(1,25) 0.0001*ones(1,25) 0.00005*ones(1,25) 0.001*ones(1,20) 0.0005*ones(1,5) 0.0001*ones(1,15) 0.00005*ones(1,15)];
    net.meta.trainOpts.learningRate = [0.01*ones(1,15)  0.005*ones(1,15) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)];
    % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    % weights completely random from Javad
    % net.meta.trainOpts.learningRate = [0.05*ones(1,10) 0.05:-0.01:0.01 0.01*ones(1,5)  0.005*ones(1,10) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,4)];
  case 'alex-net-bottle-neck'
    net.meta.trainOpts.learningRate = [0.01*ones(1,15)  0.005*ones(1,15) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)];
end

layer_lr = [.1 2] ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate);
net.meta.inputSize = [32 32 3];
net.meta.trainOpts.weightDecay = 0.0001;
net.meta.trainOpts.batchSize = 100;

switch opts.networkType
  case 'test-elnaz'
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = 1;
    % [3, 3, 3, 64]
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 3, 64, 1, 5/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    % [3, 3, 64, 64]
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 64, 64, 1, 5/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % FULLY CONNECTED
    layerNumber = layerNumber + 2;
    % [1, 1, 64, 10]
    net.layers{end+1} = convLayerRandom(layerNumber, 1, 64, 10, 0, 5/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % Loss layer
    net.layers{end+1} = struct('type', 'softmaxloss');
  case 'test';
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 3, 64, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 64, 64, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 64, 128, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 128, 128, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 128, 128, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % FULLY CONNECTED
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 8, 128, 256, 0, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 1, 256, 10, 0, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % Loss layer
    net.layers{end+1} = struct('type', 'softmaxloss');

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
  case 'alex-net'
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = 1;
    % [5, 5, 3, 96]
    % net.layers{end+1} = convLayerRandomFromPretrain(layerNumber, 2, 'load', '2D-super');
    % net.layers{end+1} = convLayerRandomFromPretrain(layerNumber, 2, 'load', '1D');
    % net.layers{end+1} = convLayerRandom(layerNumber, 5, 3, 96, 2, 5/1000, layer_lr,  'gen');
    net.layers{end+1} = convLayerRandom(layerNumber, 5, 3, 96, 2, 5/1000, 'load');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    % [5, 5, 96, 256]
    % net.layers{end+1} = convLayerRandomFromPretrain(layerNumber, 2, 'load', '2D-super');
    % net.layers{end+1} = convLayerRandomFromPretrain(layerNumber, 2, 'load', '1D');
    % net.layers{end+1} = convLayerRandom(layerNumber, 5, 96, 256, 2, 5/1000, layer_lr,  'gen');
    net.layers{end+1} = convLayerRandom(layerNumber, 5, 96, 256, 2, 5/1000, 'load');
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayerAlexNet(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 3;
    % [3, 3, 256, 384]
    % net.layers{end+1} = convLayerRandomFromPretrain(layerNumber, 1, 'load', '2D-super');
    % net.layers{end+1} = convLayerRandomFromPretrain(layerNumber, 1, 'load', '1D');
    % net.layers{end+1} = convLayerRandom(layerNumber, 3, 256, 384, 1, 5/1000, layer_lr,  'gen');
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 256, 384, 1, 5/1000, 'load');
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayerAlexNet(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 3;
    % [3, 3, 384, 384]
    % net.layers{end+1} = convLayerRandomFromPretrain(layerNumber, 1, 'load', '2D-super');
    % net.layers{end+1} = convLayerRandomFromPretrain(layerNumber, 1, 'load', '1D');
    % net.layers{end+1} = convLayerRandom(layerNumber, 3, 384, 384, 1, 5/1000, layer_lr,  'gen');
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 384, 384, 1, 5/1000, 'load');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    % [3, 3, 384, 256]
    % net.layers{end+1} = convLayerRandomFromPretrain(layerNumber, 1, 'load', '2D-super');
    % net.layers{end+1} = convLayerRandomFromPretrain(layerNumber, 1, 'load', '1D');
    % net.layers{end+1} = convLayerRandom(layerNumber, 3, 384, 256, 1, 5/1000, layer_lr,  'gen');
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 384, 256, 1, 5/1000, 'load');
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayerAlexNet(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % FULLY CONNECTED
    layerNumber = layerNumber + 3;
    % [4, 4, 256, 128]
    % net.layers{end+1} = convLayerRandom(layerNumber, 4, 256, 128, 0, 5/1000, layer_lr,  'gen');
    net.layers{end+1} = convLayerRandom(layerNumber, 4, 256, 128, 0, 5/1000, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    % [1, 1, 128, 64]
    % net.layers{end+1} = convLayerRandom(layerNumber, 1, 128, 64, 0, 5/100, layer_lr, 'gen');
    net.layers{end+1} = convLayerRandom(layerNumber, 1, 128, 64, 0, 5/100, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    % [1, 1, 64, 10]
    % net.layers{end+1} = convLayerRandom(layerNumber, 1, 64, 10, 0, 5/100, .1 * layer_lr, 'gen');
    net.layers{end+1} = convLayerRandom(layerNumber, 1, 64, 10, 0, 5/100, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % Loss layer
    net.layers{end+1} = struct('type', 'softmaxloss');
  case 'alex-net-bottle-neck'
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % k = [1,4,8,16,32]
    k = 4;
    layerNumber = 1;
    % [5, 5, 3, 96]
    net.layers{end+1} = convLayerRandom(layerNumber, 5, 3, 96, 2, 5/1000, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    % [5, 5, 96, 256]
    net.layers{end+1} = convLayerRandom(layerNumber, 5, 96, 96/k, 2, 5/1000, 'gen');
    net.layers{end+1} = convLayerRandom(layerNumber, 5, 96/k, 256, 2, 5/1000, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayerAlexNet(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 3;
    % [3, 3, 256, 384]
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 256, 256/k, 1, 5/1000, 'gend');
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 256/k, 384, 1, 5/1000, 'gend');
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayerAlexNet(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 3;
    % [3, 3, 384, 384]
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 384, 384/k, 1, 5/1000, 'gend');
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 384/k, 384, 1, 5/1000, 'gend');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    % [3, 3, 384, 256]
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 384, 384/k, 1, 5/1000, 'gend');
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 384/k, 256, 1, 5/1000, 'gend');
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayerAlexNet(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % FULLY CONNECTED
    layerNumber = layerNumber + 3;
    % [4, 4, 256, 128]
    % net.layers{end+1} = convLayerRandom(layerNumber, 4, 256, 128, 0, 5/1000, layer_lr,  'gen');
    net.layers{end+1} = convLayerRandom(layerNumber, 4, 256, 128, 0, 5/1000, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    % [1, 1, 128, 64]
    % net.layers{end+1} = convLayerRandom(layerNumber, 1, 128, 64, 0, 5/100, layer_lr, 'gen');
    net.layers{end+1} = convLayerRandom(layerNumber, 1, 128, 64, 0, 5/100, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    % [1, 1, 64, 10]
    % net.layers{end+1} = convLayerRandom(layerNumber, 1, 64, 10, 0, 5/100, .1 * layer_lr, 'gen');
    net.layers{end+1} = convLayerRandom(layerNumber, 1, 64, 10, 0, 5/100, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % Loss layer
    net.layers{end+1} = struct('type', 'softmaxloss');
  case 'mnist'
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 1, 64, 1, 1/100, layer_lr, 'gen');

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 64, 64, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = poolingLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 64, 128, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = poolingLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % FULLY CONNECTED
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 7, 128, 10, 0, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % Loss layer
    net.layers{end+1} = struct('type', 'softmaxloss');

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
  case 'vgg-16'
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 3, 64, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 64, 64, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 64, 128, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 128, 128, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 128, 256, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 256, 256, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 256, 512, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 512, 512, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 512, 512, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 512, 512, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % FULLY CONNECTED
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 8, 512, 1024, 0, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 1, 1024, 1024, 0, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 1, 1024, 10, 0, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % Loss layer
    net.layers{end+1} = struct('type', 'softmaxloss');

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
  case 'vgg-large'
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 3, 64, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 64, 64, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 64, 128, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 128, 128, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 128, 256, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 256, 256, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 256, 256, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 256, 512, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 512, 512, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 512, 512, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 512, 512, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 512, 512, 1, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 3, 512, 512, 1, 1/100, layer_lr, 'gen');


    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % FULLY CONNECTED
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 8, 512, 4096, 0, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 1, 4096, 4096, 0, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 1;
    net.layers{end+1} = convLayerRandom(layerNumber, 1, 4096, 10, 0, 1/100, layer_lr, 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % Loss layer
    net.layers{end+1} = struct('type', 'softmaxloss');

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
end

% Fill in default values
% net = vl_simplenn_tidy(net);

% --------------------------------------------------------------------
% function structuredLayer = convLayerRandom(convLayerNumber, k, m, n, pad, init_multiplier, layer_lr, source)
function structuredLayer = convLayerRandom(convLayerNumber, k, m, n, pad, init_multiplier, source)
  % source = {'load' | 'gen'}
% --------------------------------------------------------------------
  if strcmp(source, 'load')
    randomWeights = loadWeights(convLayerNumber, 'random');
  else
    randomWeights{1} = init_multiplier * randn(k, k, m, n, 'single');
    randomWeights{2} = zeros(1, n, 'single');
  end
  % structuredLayer = constructConvLayer(convLayerNumber, randomWeights, pad, layer_lr);
  structuredLayer = constructConvLayer(convLayerNumber, randomWeights, pad);

% --------------------------------------------------------------------
function structuredLayer = convLayerBaseline(convLayerNumber, pad)
  % pre-trained by javad, then trained 3, 5 or 8 more epochs by amir.
% --------------------------------------------------------------------
  baselineWeights = loadWeights(convLayerNumber, 'baseline');
  structuredLayer = constructConvLayer(convLayerNumber, baselineWeights, pad);

% --------------------------------------------------------------------
function structuredLayer = convLayerRandomFromPretrain(convLayerNumber, pad, source, random_gen_type)
  % source = {'load' | 'gen'}
  % random_gen_type = {'1D', '2D-super'}
% --------------------------------------------------------------------
  if strcmp(source, 'load')
    if strcmp(random_gen_type, '1D') || strcmp(random_gen_type, '2D-super')
      folder_name = sprintf('random-from-baseline-%s', random_gen_type);
      randomWeights = loadWeights(convLayerNumber, folder_name);
    else
      throwException('unrecognized command');
    end
  elseif strcmp(source, 'gen')
    utils = networkExtractionUtils;
    baselineWeights = loadWeights(convLayerNumber, 'baseline'); % used for its size
    if strcmp(random_gen_type, '1D')
      randomWeights = utils.genRandomWeightsFromBaseline1DGaussian( ...
        baselineWeights, ...
        convLayerNumber);
    elseif strcmp(random_gen_type, '2D-super')
      randomWeights = utils.genRandomWeightsFromBaseline2DGaussianSuper( ...
        baselineWeights, ...
        convLayerNumber);
    else
      throwException('unrecognized command');
    end
  else
    throwException('unrecognized command');
  end
  structuredLayer = constructConvLayer(convLayerNumber, randomWeights, pad);
  % randomWeights2{1} = randomWeights{1} * .1;
  % randomWeights2{2} = randomWeights{2} * .1;
  % structuredLayer = constructConvLayer(convLayerNumber, randomWeights2, pad);

% --------------------------------------------------------------------
function weights = loadWeights(convLayerNumber, weightType)
  % weightType = {
  %   'random' |
  %   'baseline' |
  %   'random-from-baseline-1D' |
  %   'random-from-baseline-2D-super'
  % }
% --------------------------------------------------------------------
  fprintf( ...
    '[INFO] Loading %s weights (layer %d) from saved directory...\t', ...
    weightType, ...
    convLayerNumber);
  devPath = getDevPath();
  subDirPath = fullfile('data', 'cifar-alexnet', sprintf('+8epoch-%s', weightType));
  fileNameSuffix = sprintf('-layer-%d.mat', convLayerNumber);
  tmp = load(fullfile(devPath, subDirPath, sprintf('W1%s', fileNameSuffix)));
  weights{1} = tmp.W1;
  tmp = load(fullfile(devPath, subDirPath, sprintf('W2%s', fileNameSuffix)));
  weights{2} = tmp.W2;
  fprintf('Done!\n');

% --------------------------------------------------------------------
% function structuredLayer = constructConvLayer(index, weights, pad, layer_lr)
function structuredLayer = constructConvLayer(index, weights, pad)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'conv', ...
    'name', sprintf('conv%s', index), ...
    'weights', {weights}, ...
    'stride', 1, ...
    'pad', pad);
    % 'learningRate', layer_lr, ...

% --------------------------------------------------------------------
function structuredLayer = reluLayer(convLayerNumber)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'relu', ...
    'name', sprintf('relu%s', convLayerNumber));

% --------------------------------------------------------------------
function structuredLayer = poolingLayer(convLayerNumber)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'pool', ...
    'name', sprintf('pool%s', convLayerNumber), ...
    'method', 'max', ...
    'pool', [2 2], ...
    'stride', 2, ...
    'pad', 0); % Emulate caffe

% --------------------------------------------------------------------
function structuredLayer = poolingLayerAlexNet(convLayerNumber)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'pool', ...
    'name', sprintf('pool%s', convLayerNumber), ...
    'method', 'max', ...
    'pool', [3 3], ...
    'stride', 2, ...
    'pad', [0 1 0 1]); % Emulate caffe

% --------------------------------------------------------------------
function throwException(msg)
% --------------------------------------------------------------------
  msgID = 'MYFUN:BadIndex';
  throw(MException(msgID,msg));

