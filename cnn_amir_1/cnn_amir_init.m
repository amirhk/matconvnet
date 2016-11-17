function net = cnn_amir_init(varargin)
opts.networkArch = 'alexnet';
opts.dataset = 'cifar';
opts.backpropDepth = 20;
opts.weightDecay = 0.0001;
opts.weightInitSequence = {'compRand', 'compRand', 'compRand', 'compRand', 'compRand'};
opts.weightInitSource = 'gen';
opts.bottleneckDivideBy = 1;
opts = vl_argparse(opts, varargin);

tic;
rng(0);
net.layers = {};
% Meta parameters
switch opts.networkArch
  case 'mnistnet'
    switch opts.dataset
      case 'mnist'
        net.meta.trainOpts.learningRate = [0.001*ones(1,50)]; % matconvnet default
    end
  case 'lenet'
    switch opts.dataset
      case 'cifar'
        net.meta.trainOpts.learningRate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)]; % matconvnet default
      case 'coil-100'
        % testing
        net.meta.trainOpts.learningRate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)]; % matconvnet default
      case 'stl-10'
        net.meta.trainOpts.learningRate = [0.5*ones(1,20) 0.05*ones(1,15)  0.1:-0.01:0.06 0.05*ones(1,10)]; % javad-LR
    end
  case 'alexnet'
    switch opts.dataset
      case 'cifar'
        net.meta.trainOpts.learningRate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)]; % amir-LR
      case 'stl-10'
        % still testing
        % net.meta.trainOpts.learningRate = [0.5*ones(1,20) 0.05*ones(1,15)  0.1:-0.01:0.06 0.05*ones(1,10)]; % testing
        % net.meta.trainOpts.learningRate = [2*ones(1,20)]; % testing
        net.meta.trainOpts.learningRate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)]; % amir-LR
    end
  case 'alexnet-bnorm'
    net.meta.trainOpts.learningRate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)];
  case 'alexnet-bottleneck'
    net.meta.trainOpts.learningRate = [0.005*ones(1,50)];
end

net.meta.trainOpts.weightInitSequence = printWeightInitSequence(opts.weightInitSequence);
net.meta.trainOpts.weightInitSource = opts.weightInitSource;
net.meta.trainOpts.backpropDepth = opts.backpropDepth;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate);
net.meta.inputSize = [32 32 3];
net.meta.trainOpts.weightDecay = opts.weightDecay;
net.meta.trainOpts.batchSize = 100;
opts = vl_argparse(opts, varargin);

switch opts.networkArch
  case 'mnistnet'
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- ---                                                     --- --- --
    % --- --- ---                    MNISTNET                         --- --- --
    % --- --- ---                                                     --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = 1;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 5, 1, 20, 1/100, 0, char(opts.weightInitSequence{1}), opts.weightInitSource);
    net.layers{end+1} = poolingLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 5, 20, 50, 1/100, 0, char(opts.weightInitSequence{1}), opts.weightInitSource);
    net.layers{end+1} = poolingLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 4, 50, 500, 1/100, 0, char(opts.weightInitSequence{1}), opts.weightInitSource);
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 1, 500, 10, 1/100, 0, 'compRand', 'gen');

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % Loss layer
    net.layers{end+1} = struct('type', 'softmaxloss');
  case 'lenet'
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- ---                                                     --- --- --
    % --- --- ---                     LENET                           --- --- --
    % --- --- ---                                                     --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = 1;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 5, 3, 32, 1/100, 2, char(opts.weightInitSequence{1}), opts.weightInitSource);
    net.layers{end+1} = poolingLayerLeNetMax(layerNumber);
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = dropoutLayer(layerNumber, 0.5); % NEW!!!!!!!!!

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 3;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 5, 32, 32, 5/100, 2, char(opts.weightInitSequence{2}), opts.weightInitSource);
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayerLeNetAvg(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 3;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 5, 32, 64, 5/100, 2, char(opts.weightInitSequence{3}), opts.weightInitSource);
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayerLeNetAvg(layerNumber);
    net.layers{end+1} = dropoutLayer(layerNumber, 0.5); % NEW!!!!!!!!!

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % FULLY CONNECTED
    layerNumber = layerNumber + 3;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 4, 64, 64, 5/100, 0, 'compRand', 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 1, 64, 10, 5/100, 0, 'compRand', 'gen');
    % net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 1, 64, 100, 5/100, 0, 'compRand', 'gen');

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % Loss layer
    net.layers{end+1} = struct('type', 'softmaxloss');
  case 'alexnet'
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- ---                                                     --- --- --
    % --- --- ---                   ALEXNET                           --- --- --
    % --- --- ---                                                     --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = 1;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 5, 3, 96, 5/1000, 2, char(opts.weightInitSequence{1}), opts.weightInitSource);
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 5, 96, 256, 5/1000, 2, char(opts.weightInitSequence{2}), opts.weightInitSource);
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayerAlexNet(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 3;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 3, 256, 384, 5/1000, 1, char(opts.weightInitSequence{3}), opts.weightInitSource);
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayerAlexNet(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 3;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 3, 384, 384, 5/1000, 1, char(opts.weightInitSequence{4}), opts.weightInitSource);
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 3, 384, 256, 5/1000, 1, char(opts.weightInitSequence{5}), opts.weightInitSource);
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayerAlexNet(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % FULLY CONNECTED
    layerNumber = layerNumber + 3;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 4, 256, 128, 5/1000, 0, 'compRand', 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 1, 128, 64, 5/100, 0, 'compRand', 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 1, 64, 10, 5/100, 0, 'compRand', 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % Loss layer
    net.layers{end+1} = struct('type', 'softmaxloss');
  case 'alexnet-bnorm'
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- ---                                                     --- --- --
    % --- --- ---                ALEXNET-BNORM                        --- --- --
    % --- --- ---                                                     --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = 1;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 5, 3, 96, 5/1000, 2, char(opts.weightInitSequence{1}), opts.weightInitSource);
    % net.layers{end+1} = bnormLayer(layerNumber, 96);
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 5, 96, 256, 5/1000, 2, char(opts.weightInitSequence{2}), opts.weightInitSource);
    net.layers{end+1} = bnormLayer(layerNumber, 256);
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayerAlexNet(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 3;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 3, 256, 384, 5/1000, 1, char(opts.weightInitSequence{3}), opts.weightInitSource);
    % net.layers{end+1} = bnormLayer(layerNumber, 384);
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayerAlexNet(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 3;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 3, 384, 384, 5/1000, 1, char(opts.weightInitSequence{4}), opts.weightInitSource);
    % net.layers{end+1} = bnormLayer(layerNumber, 384);
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 3, 384, 256, 5/1000, 1, char(opts.weightInitSequence{5}), opts.weightInitSource);
    net.layers{end+1} = bnormLayer(layerNumber, 256);
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayerAlexNet(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % FULLY CONNECTED
    layerNumber = layerNumber + 3;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 4, 256, 128, 5/1000, 0, 'compRand', 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 1, 128, 64, 5/100, 0, 'compRand', 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 1, 64, 10, 5/100, 0, 'compRand', 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % Loss layer
    net.layers{end+1} = struct('type', 'softmaxloss');
  case 'alexnet-bottleneck'
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % --- --- ---                                                     --- --- --
    % --- --- ---               ALEXNET-BOTTLENECK                    --- --- --
    % --- --- ---                                                     --- --- --
    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % k = [1,4,8,16,32]
    k = opts.bottleneckDivideBy;
    layerNumber = 1;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 5, 3, 96, 5/1000, 2, 'compRand', 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    % net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 5, 96, 256, 5/1000, 2, 'compRand', 'gen');
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 5, 96, 96/k, 5/1000, 2, 'compRand', 'gen');
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 5, 96/k, 256, 5/1000, 2, 'compRand', 'gen');
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayerAlexNet(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 3;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 3, 256, 384, 5/1000, 1, 'compRand', 'gen');
    % net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 3, 256, 256/k, 5/1000, 1, 'compRand', 'gen');
    % net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 3, 256/k, 384, 5/1000, 1, 'compRand', 'gen');
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayerAlexNet(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 3;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 3, 384, 384, 5/1000, 1, 'compRand', 'gen');
    % net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 3, 384, 384/k, 5/1000, 1, 'compRand', 'gen');
    % net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 3, 384/k, 384, 5/1000, 1, 'compRand', 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 3, 384, 256, 5/1000, 1, 'compRand', 'gen');
    % net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 3, 384, 384/k, 5/1000, 1, 'compRand', 'gen');
    % net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 3, 384/k, 256, 5/1000, 1, 'compRand', 'gen');
    net.layers{end+1} = reluLayer(layerNumber);
    net.layers{end+1} = poolingLayerAlexNet(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % FULLY CONNECTED
    layerNumber = layerNumber + 3;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 4, 256, 128, 5/1000, 0, 'compRand', 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 1, 128, 64, 5/100, 0, 'compRand', 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    layerNumber = layerNumber + 2;
    net.layers{end+1} = convLayer(opts.dataset, opts.networkArch, layerNumber, 1, 64, 10, 5/100, 0, 'compRand', 'gen');
    net.layers{end+1} = reluLayer(layerNumber);

    % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    % Loss layer
    net.layers{end+1} = struct('type', 'softmaxloss');
end

% --------------------------------------------------------------------
function structuredLayer = convLayer(dataset, networkArch, layerNumber, k, m, n, init_multiplier, pad, weightInitType, weightInitSource);
% --------------------------------------------------------------------
  switch weightInitSource
    case 'load'
      layerWeights = loadWeights(dataset, networkArch, layerNumber, weightInitType);
    case 'gen'
      if ~strcmp(weightInitType, 'compRand')
        utils = networkExtractionUtils;
        baselineWeights = loadWeights(dataset, networkArch, layerNumber, 'baseline'); % used for its size
      end
      switch weightInitType
        case 'compRand'
          layerWeights{1} = init_multiplier * randn(k, k, m, n, 'single');
          layerWeights{2} = zeros(1, n, 'single');
        otherwise
          throwException('[ERROR] Generating non-compRand weights not supported from this code.');
      end
  end
  structuredLayer = constructConvLayer(networkArch, layerNumber, layerWeights, pad, weightInitType, weightInitSource);

% --------------------------------------------------------------------
function weights = loadWeights(dataset, networkArch, layerNumber, weightInitType)
% --------------------------------------------------------------------
  fprintf( ...
    '[INFO] Loading %s weights (layer %d) from saved directory...\t', ...
    weightInitType, ...
    layerNumber);
  devPath = getDevPath();

  % subDirPath = fullfile('data', 'cifar-alexnet', sprintf('w_%s', weightInitType));
  % TODO: search subtstring... if networkArch starts with 'alexnet' use the 'alexnet' folder
  subDirPath = fullfile('data', sprintf('%s-%s', dataset, networkArch), sprintf('w_%s', weightInitType));
  fileNameSuffix = sprintf('-layer-%d.mat', layerNumber);
  tmp = load(fullfile(devPath, subDirPath, sprintf('W1%s', fileNameSuffix)));
  weights{1} = tmp.W1;
  tmp = load(fullfile(devPath, subDirPath, sprintf('W2%s', fileNameSuffix)));
  weights{2} = tmp.W2;
  fprintf('Done!\n');

% --------------------------------------------------------------------
function structuredLayer = constructConvLayer(networkArch, layerNumber, weights, pad, weightInitType, weightInitSource)
% --------------------------------------------------------------------
  lr = [.1 2];
  if strcmp(networkArch, 'alexnet') && layerNumber == 18
    lr = lr * .1;
  elseif strcmp(networkArch, 'lenet') && layerNumber == 12
  end
  structuredLayer = struct( ...
    'type', 'conv', ...
    'name', sprintf('conv%s-%s-%s', layerNumber, weightInitType, weightInitSource), ...
    'weights', {weights}, ...
    'learningRate', lr, ...
    'stride', 1, ...
    'pad', pad);

% --------------------------------------------------------------------
function structuredLayer = reluLayer(layerNumber)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'relu', ...
    'name', sprintf('relu%s', layerNumber));

% --------------------------------------------------------------------
function structuredLayer = poolingLayer(layerNumber)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'pool', ...
    'name', sprintf('pool%s', layerNumber), ...
    'method', 'max', ...
    'pool', [2 2], ...
    'stride', 2, ...
    'pad', 0); % Emulate caffe

% --------------------------------------------------------------------
function structuredLayer = poolingLayerAlexNet(layerNumber)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'pool', ...
    'name', sprintf('pool%s', layerNumber), ...
    'method', 'max', ...
    'pool', [3 3], ...
    'stride', 2, ...
    'pad', [0 1 0 1]); % Emulate caffe

% --------------------------------------------------------------------
function structuredLayer = poolingLayerLeNetAvg(layerNumber)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'pool', ...
    'name', sprintf('pool%s', layerNumber), ...
    'method', 'avg', ...
    'pool', [3 3], ...
    'stride', 2, ...
    'pad', [0 1 0 1]); % Emulate caffe

% --------------------------------------------------------------------
function structuredLayer = poolingLayerLeNetMax(layerNumber)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'pool', ...
    'name', sprintf('pool%s', layerNumber), ...
    'method', 'max', ...
    'pool', [3 3], ...
    'stride', 2, ...
    'pad', [0 1 0 1]); % Emulate caffe

% --------------------------------------------------------------------
function structuredLayer = dropoutLayer(layerNumber, dropoutRatio)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'dropout', ...
    'name', sprintf('dropout%s', layerNumber), ...
    'rate', dropoutRatio);

% --------------------------------------------------------------------
function structuredLayer = bnormLayer(layerNumber, ndim)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'bnorm', ...
    'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
    'learningRate', [1 1], ...
    'weightDecay', [0 0]);

% --------------------------------------------------------------------
function throwException(msg)
% --------------------------------------------------------------------
  msgID = 'MYFUN:BadIndex';
  throw(MException(msgID,msg));
