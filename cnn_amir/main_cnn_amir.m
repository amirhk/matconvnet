function main_cnn_amir(varargin)
  whitenData = true;
  contrastNormalization = true;
  % backpropDepthList = [7, 10, 12, 15, 18, 20];
  backpropDepthList = [20, 18, 15, 12, 10, 7];
  % backpropDepthList = [20];
  for backpropDepth = backpropDepthList
    cnn_amir( ...
      'whitenData', whitenData, ...
      'contrastNormalization', contrastNormalization, ...
      'backpropDepth', backpropDepth);
  end
