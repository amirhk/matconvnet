function main_cnn_amir(varargin)
  folder = fileparts(mfilename('fullpath'));
  folderNumber = str2num(folder(end));

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                   -- ==
% -- ==                        NETWORK ARCH                               -- ==
% -- ==                                                                   -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

  % networkArch = 'lenet';
  % % backpropDepthList = [13, 10, 7, 4];
  % backpropDepthList = [13];
  % bottleneckDivideByList = [1];

  networkArch = 'alexnet';
  backpropDepthList = [20, 18, 15, 12, 10, 7];
  % backpropDepthList = [20];
  bottleneckDivideByList = [1];

  % networkArch = 'alexnet-bnorm';
  % % backpropDepthList = [20, 18, 15, 12, 10, 7];
  % backpropDepthList = [22];
  % bottleneckDivideByList = [1];

  % networkArch = 'alexnet-bottleneck';
  % backpropDepthList = [21];
  % bottleneckDivideByList = [1,2,4,8,16,32];

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                   -- ==
% -- ==                          MORE PARAMS                              -- ==
% -- ==                                                                   -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

  dataset = 'cifar';
  weightInitSource = 'load';  % {'load' | 'gen'}

  % weightInitTypeList = {'baseline', 'compRand', '1D', '2D-positive', '2D-super', '2D-posneg', '2D-shiftflip', '2D-mult-randn', '2D-mult-kernel'};
  weightInitTypeList = {'1D'};

  % imdbPortionList = [0.1, 0.25, 0.5, 1.0];
  imdbPortionList = [1.0];

  % weightDecayList = [0.1, 0.01, 0.001, 0.0001, 0]; % Works: {0.001, 0.0001, 0} Doesn't Work: {0.1, 0.01}
  weightDecayList = [0.0001];

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                   -- ==
% -- ==                           MAIN LOOP                               -- ==
% -- ==                                                                   -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

  for weightInitType = weightInitTypeList
    for bottleneckDivideBy = bottleneckDivideByList
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
              'bottleneckDivideBy', bottleneckDivideBy);
          end
        end
      end
    end
  end
