function main_cnn_amir(varargin)
  weightInitType = 'compRand';
  weightInitSource = 'load';
  backpropDepthList = [20, 18, 15, 12, 10, 7];
  % backpropDepthList = [20];
  for backpropDepth = backpropDepthList
    cnn_amir( ...
      'weightInitType', weightInitType, ...
      'weightInitSource', weightInitSource, ...
      'backpropDepth', backpropDepth);
  end
