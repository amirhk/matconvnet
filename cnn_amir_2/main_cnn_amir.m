function main_cnn_amir(varargin)
  folder = fileparts(mfilename('fullpath'));
  folderNumber = str2num(folder(end));

  networkArch = 'lenet';
  % backpropDepthList = [13, 10, 7, 4];
  backpropDepthList = [13];
  bottleNeckDivideByList = [1];

  % networkArch = 'alex-net';
  % backpropDepthList = [20, 18, 15, 12, 10, 7];
  % % backpropDepthList = [20];
  % bottleNeckDivideByList = [1];

  % networkArch = 'alex-net-bnorm';
  % % backpropDepthList = [20, 18, 15, 12, 10, 7];
  % backpropDepthList = [22];
  % bottleNeckDivideByList = [1];

  % networkArch = 'alex-net-bottle-neck';
  % backpropDepthList = [21];
  % bottleNeckDivideByList = [1,2,4,8,16,32];

  dataset = 'cifar';
  weightInitSource = 'load';
  % weightInitType = {'compRand'};
  weightInitTypeList = {'compRand', '1D', '2D-mult', '2D-super', '2D-pos-neg'};

  % imdbPortionList = [0.1, 0.25, 0.5, 1.0];
  imdbPortionList = [1.0];

  % weightDecayList = [0.1, 0.01, 0.001, 0.0001, 0];
  weightDecayList = [0.0001];
  for weightInitType = weightInitTypeList
    for bottleNeckDivideBy = bottleNeckDivideByList
      for imdbPortion = imdbPortionList
        for weightDecay = weightDecayList
          for backpropDepth = backpropDepthList
            cnn_amir( ...
              'folderNumber', folderNumber, ...
              'networkArch', networkArch, ...
              'dataset', dataset, ...
              'imdbPortion', imdbPortion, ...
              'backpropDepth', backpropDepth, ...
              'weightDecay', weightDecay, ...
              'weightInitType', char(weightInitType), ...
              'weightInitSource', weightInitSource, ...
              'bottleNeckDivideBy', bottleNeckDivideBy);
          end
        end
      end
    end
  end
