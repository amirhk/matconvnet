function main_cnn_amir(varargin)
  % networkArch = 'alex-net';
  % backpropDepthList = [20, 18, 15, 12, 10, 7];
  % backpropDepthList = [20];

  networkArch = 'alex-net-bottle-neck';
  backpropDepthList = [21];
  bottleNeckDivideByList = [1,2,4,8,16,32];
  % bottleNeckDivideByList = [1];

  dataset = 'cifar';
  weightInitType = 'compRand';
  weightInitSource = 'load';

  % imdbPortionList = [0.1, 0.25, 0.5, 1.0];
  imdbPortionList = [1.0];

  % weightDecayList = [0.1, 0.01, 0.001, 0.0001, 0];
  weightDecayList = [0.0001];

  for bottleNeckDivideBy = bottleNeckDivideByList
    for imdbPortion = imdbPortionList
      for weightDecay = weightDecayList
        for backpropDepth = backpropDepthList
          cnn_amir( ...
            'networkArch', networkArch, ...
            'dataset', dataset, ...
            'imdbPortion', imdbPortion, ...
            'backpropDepth', backpropDepth, ...
            'weightDecay', weightDecay, ...
            'weightInitType', weightInitType, ...
            'weightInitSource', weightInitSource, ...
            'bottleNeckDivideBy', bottleNeckDivideBy);
        end
      end
    end
  end
