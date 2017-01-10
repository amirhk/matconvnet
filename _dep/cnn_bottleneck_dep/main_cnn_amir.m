function main_cnn_amir(varargin)

  for num_bottle_necks = [1,2]
  % for num_bottle_necks = [2]
    for k = [16,8,4,2,1]
    % for k = [1]
      cnn_cifar( ...
        'num_bottle_necks', num_bottle_necks, ...
        'k_amir', k);
    end
  end
  % folder = fileparts(mfilename('fullpath'));
  % folderNumber = str2num(folder(end));

  % networkArch = 'alex-net';
  % % backpropDepthList = [20, 18, 15, 12, 10, 7];
  % backpropDepthList = [20];
  % bottleNeckDivideByList = [1];

  % % networkArch = 'alex-net-bnorm';
  % % % backpropDepthList = [20, 18, 15, 12, 10, 7];
  % % backpropDepthList = [22];
  % % bottleNeckDivideByList = [1];

  % % networkArch = 'alex-net-bottle-neck';
  % % backpropDepthList = [21];
  % % bottleNeckDivideByList = [1,2,4,8,16,32];

  % dataset = 'cifar';
  % weightInitType = '1D';
  % weightInitSource = 'gen';

  % % imdbPortionList = [0.1, 0.25, 0.5, 1.0];
  % imdbPortionList = [1.0];

  % % weightDecayList = [0.1, 0.01, 0.001, 0.0001, 0];
  % weightDecayList = [0.0001];

  % for bottleNeckDivideBy = bottleNeckDivideByList
  %   for imdbPortion = imdbPortionList
  %     for weightDecay = weightDecayList
  %       for backpropDepth = backpropDepthList
  %         cnn_amir( ...
  %           'folderNumber', folderNumber, ...
  %           'networkArch', networkArch, ...
  %           'dataset', dataset, ...
  %           'imdbPortion', imdbPortion, ...
  %           'backpropDepth', backpropDepth, ...
  %           'weightDecay', weightDecay, ...
  %           'weightInitType', weightInitType, ...
  %           'weightInitSource', weightInitSource, ...
  %           'bottleNeckDivideBy', bottleNeckDivideBy);
  %       end
  %     end
  %   end
  % end
