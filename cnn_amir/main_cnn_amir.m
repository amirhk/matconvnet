function main_cnn_amir(varargin)
  networkType = 'alex-net-bottle-neck';
  dataset = 'cifar';
  % weightDecayList = [0.1, 0.01, 0.001, 0.0001, 0];
  weightDecayList = [0.0001];
  weightInitType = 'compRand';
  weightInitSource = 'load';
  % backpropDepthList = [20, 18, 15, 12, 10, 7];
  backpropDepthList = [21];
  for weightDecay = weightDecayList
    for backpropDepth = backpropDepthList
      cnn_amir( ...
        'networkType', networkType, ...
        'dataset', dataset, ...
        'weightDecay', weightDecay, ...
        'weightInitType', weightInitType, ...
        'weightInitSource', weightInitSource, ...
        'backpropDepth', backpropDepth);
    end
  end
