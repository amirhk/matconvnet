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
  case 'test'; % net.meta.trainOpts.learningRate = [0.001 * ones(1,5) 0.01 * ones(1,10)];
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
