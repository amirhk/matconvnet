function main_cnn_amir(varargin)
  weightDecayList = [0.1, 0.01, 0.001, 0.0001, 0];
  weightInitType = '2D';
  weightInitSource = 'load';
  % backpropDepthList = [20, 18, 15, 12, 10, 7];
  backpropDepthList = [20];
  for weightDecay = weightDecayList
    for backpropDepth = backpropDepthList
      cnn_amir( ...
        'weightDecay', weightDecay, ...
        'weightInitType', weightInitType, ...
        'weightInitSource', weightInitSource, ...
        'backpropDepth', backpropDepth);
    end
  end
