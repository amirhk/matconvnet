
function main_cnn_amir(varargin)
  folder = fileparts(mfilename('fullpath'));
  folderNumber = str2num(folder(end));

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                   -- ==
% -- ==                        NETWORK ARCH                               -- ==
% -- ==                                                                   -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

  dataset = 'coil-100'; % {'mnist', 'cifar', 'stl-10', 'coil-100'}

  % networkArch = 'mnistnet';
  % % backpropDepthList = [8, 6, 4];
  % backpropDepthList = [4];

  networkArch = 'lenet';
  backpropDepthList = [13, 10, 7, 4];
  % backpropDepthList = [13];

  % networkArch = 'lenet';
  % % backpropDepthList = [13, 10, 7, 4];
  % backpropDepthList = [13];

  % networkArch = 'alexnet';
  % % backpropDepthList = [20, 18, 15, 12, 10, 7];
  % backpropDepthList = [20];

  % networkArch = 'alexnet-bnorm';
  % % backpropDepthList = [20, 18, 15, 12, 10, 7];
  % backpropDepthList = [22];

  % networkArch = 'alexnet-bottleneck';
  % backpropDepthList = [21];
  % bottleneckDivideByList = [1,2,4,8,16,32];
  bottleneckDivideByList = [1];

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                   -- ==
% -- ==                          MORE PARAMS                              -- ==
% -- ==                                                                   -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

  % weightInitSource = 'gen';  % {'load' | 'gen'}
  % weightInitSequenceList = {{'compRand', 'compRand', 'compRand', 'compRand', 'compRand'}};

  weightInitSource = 'load';  % {'load' | 'gen'}
  % % weightInitTypes: {'baseline', 'compRand', '1D', '2D-positive', '2D-super', '2D-posneg', '2D-shiftflip', '2D-mult-randn', '2D-mult-kernel'};
  % % weightInitSequenceList = {{'baseline', 'baseline', 'baseline', 'baseline', 'baseline'}};
  % % weightInitSequenceList = { ...
  % %   {'baseline', 'baseline', 'baseline', 'baseline', 'baseline'}, ...
  % %   {'compRand', 'compRand', 'compRand', 'compRand', 'compRand'}, ...
  % %   {'1D', '1D', '1D', '1D', '1D'}, ...
  % %   {'2D-shiftflip', '2D-shiftflip', '1D', '1D', '1D'}, ...
  % %   {'2D-shiftflip', '2D-shiftflip', 'compRand', 'compRand', 'compRand'}, ...
  % %   {'2D-mult-randn', '2D-mult-randn', '1D', '1D', '1D'}, ...
  % %   {'2D-mult-randn', '2D-mult-randn', 'compRand', 'compRand', 'compRand'}};ult-randn', '2D-mult-randn', 'compRand'}};
  weightInitSequenceList = { ...
    % {'baseline', 'baseline', 'baseline'}, ...
    % {'compRand', 'compRand', 'compRand'}, ...
    {'1D', '1D', '1D'}, ...
    {'2D-shiftflip', '1D', '1D'}, ...
    {'2D-shiftflip', 'compRand', 'compRand'}, ...
    {'2D-mult-randn', '1D', '1D'}, ...
    {'2D-mult-randn', 'compRand', 'compRand'}};

  % imdbPortionList = [0.1, 0.25, 0.5, 1.0];
  imdbPortionList = [1.0];

  % weightDecayList = [0.1, 0.01, 0.001, 0.0001, 0]; % Works: {0.001, 0.0001, 0} Doesn't Work: {0.1, 0.01}
  weightDecayList = [0.0001];

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                   -- ==
% -- ==                           MAIN LOOP                               -- ==
% -- ==                                                                   -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

  for weightInitSequence = weightInitSequenceList
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
              'weightInitSequence', weightInitSequence{1}, ...
              'weightInitSource', weightInitSource, ...
              'bottleneckDivideBy', bottleneckDivideBy);
          end
        end
      end
    end
  end
