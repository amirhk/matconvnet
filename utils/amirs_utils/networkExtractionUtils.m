function fh = networkExtractionUtils()
  % assign function handles so we can call these local functions from elsewhere
  fh.extractNewWeightsFromNetwork = @extractNewWeightsFromNetwork;
  fh.extractAllNewWeightsFromNetwork = @extractAllNewWeightsFromNetwork;

% --------------------------------------------------------------------
function extractNewWeightsFromNetwork(networkArch, weightInitType)
  % networkArch = {'alexnet', 'lenet'}
  % weightInitType = {'baseline', 'compRand', '1D', '2D-positive', '2D-mult', '2D-mult2', '2D-super', '2D-posneg', '2D-shiftflip'};
% --------------------------------------------------------------------
  fprintf('\n-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --\n');
  fprintf(sprintf('[INFO] Loading data from pre-trained %s on Cifar...\n', networkArch));
  devPath = getDevPath();
  loadedFile = load(fullfile( ...
    devPath, ...
    'data', ...
    sprintf('cifar-%s', networkArch), ...
    sprintf('pretrained-%s.mat', networkArch)));
  fprintf('[INFO] Loading data successful!\n\n');
  net = loadedFile.net;
  % printNetworkStructure(net);
  % TODO: change these....
  switch weightInitType
    case 'baseline'
      genWeightsMethod = @genBaselineWeights;
    case 'compRand'
      genWeightsMethod = @genCompRandWeights;
    case '1D'
      genWeightsMethod = @gen1DGaussianWeightsFromBaseline;
    case '2D-mult'
      genWeightsMethod = @gen2DGaussianMultWeightsFromBaseline;
    case '2D-mult2'
      genWeightsMethod = @gen2DGaussianMult2WeightsFromBaseline;
    case '2D-super'
      genWeightsMethod = @gen2DGaussianSuperWeightsFromBaseline;
    case '2D-posneg'
      genWeightsMethod = @gen2DGaussianPosNegWeightsFromBaseline;
    case '2D-positive'
      genWeightsMethod = @gen2DGaussianPositiveWeightsFromBaseline;
    case '2D-shiftflip'
      genWeightsMethod = @gen2DGaussianShiftFlipWeightsFromBaseline;
  end
  genNewWeights(networkArch, weightInitType, net, genWeightsMethod);

% --------------------------------------------------------------------
function extractAllNewWeightsFromNetwork(networkArch)
  % networkArch = {'alexnet', 'lenet'}
% --------------------------------------------------------------------
  runInTryCatch(@extractNewWeightsFromNetwork, networkArch, 'baseline');
  runInTryCatch(@extractNewWeightsFromNetwork, networkArch, 'compRand');
  runInTryCatch(@extractNewWeightsFromNetwork, networkArch, '1D');
  runInTryCatch(@extractNewWeightsFromNetwork, networkArch, '2D-mult');
  runInTryCatch(@extractNewWeightsFromNetwork, networkArch, '2D-mult2');
  runInTryCatch(@extractNewWeightsFromNetwork, networkArch, '2D-super');
  runInTryCatch(@extractNewWeightsFromNetwork, networkArch, '2D-posneg');
  runInTryCatch(@extractNewWeightsFromNetwork, networkArch, '2D-positive');
  runInTryCatch(@extractNewWeightsFromNetwork, networkArch, '2D-shiftflip');

% --------------------------------------------------------------------
function genNewWeights(networkArch, weightInitType, net, genWeightsMethod)
  % for all 'conv' layers...
% --------------------------------------------------------------------
  fprintf(sprintf('[INFO] Generating new `%s` weights from `%s`... \n', weightInitType, networkArch));
  layers = net.layers;
  for i = 1:numel(layers)
    if (strcmp(layers{i}.type, 'conv'))
      layerNumber = i;
      newWeights = genWeightsMethod(layers, layerNumber);
      saveNewWeights(networkArch, weightInitType, newWeights, layerNumber);
    end
  end
  fprintf('[INFO] Successfully finished generating weights!\n\n');

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                                           -- ==
% -- ==                                 RANDOM / BASELINE                                         -- ==
% -- ==                                                                                           -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

% --------------------------------------------------------------------
function randomWeights = genBaselineWeights(layers, layerNumber)
  % just save the baseline in the form of weights, not the whole network
% --------------------------------------------------------------------
  randomWeights{1} = layers{layerNumber}.weights{1};
  randomWeights{2} = layers{layerNumber}.weights{2};

% --------------------------------------------------------------------
function randomWeights = genCompRandWeights(layers, layerNumber)
  % extract ~400,000 kernels and just save that many random kernels of the same
  % size generated from randn methods
% --------------------------------------------------------------------
  W1 = layers{layerNumber}.weights{1};
  k = size(W1, 1);
  k = size(W1, 2);
  m = size(W1, 3);
  n = size(W1, 4);
  init_multiplier = 5/1000;
  randomWeights{1} = init_multiplier * randn(k, k, m, n, 'single');
  randomWeights{2} = zeros(1, n, 'single');

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                                           -- ==
% -- ==                                          1D                                               -- ==
% -- ==                                                                                           -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

% --------------------------------------------------------------------
function newWeights = gen1DGaussianWeightsFromBaseline( ...
  layers, ...
  layerNumber)
  % extract ~400,000 kernels, fit dist to each, and generate new kernel for each
% --------------------------------------------------------------------
  tic;
  saveDistributions = false;
  baselineKernelDists = getDistsFromBaseline1DGaussian( ...
    layers, ...
    layerNumber, ...
    saveDistributions);
  newWeights{1} = gennewWeightsFromDistributions1DGaussian( ...
    layers, ...
    layerNumber, ...
    baselineKernelDists);
  % Don't use a dist, just put the baseline pretrain
  newWeights{2} = layers{layerNumber}.weights{2};
  numberOfKernels = size(newWeights{1}, 3) * size(newWeights{1}, 4);
  fprintf( ...
    ['[INFO] layer %d: 1D weights generated for %d kernels. ', ...
    'Elapsed Time (since start): %f\n'], ...
    layerNumber, ...
    numberOfKernels, ...
    toc);

% --------------------------------------------------------------------
function baselineKernelDists = getDistsFromBaseline1DGaussian( ...
  layers, ...
  layerNumber, ...
  saveDistributions)
% --------------------------------------------------------------------
  baselineWeights = layers{layerNumber}.weights{1};
  baselineKernelDists = [];
  for d = 1:size(baselineWeights, 4)
    for c = 1:size(baselineWeights, 3)
      baselineKernel = baselineWeights(:, :, c, d);
      matrixDims = [size(baselineKernel, 1), size(baselineKernel, 2)];
      vectorDims = [size(baselineKernel, 1) * size(baselineKernel, 2), 1];
      vectorizedBaselineKernel = reshape(baselineKernel, vectorDims);
      dist = fitdist(vectorizedBaselineKernel, 'Normal');
      baselineKernelDists(:, :, c, d) = [dist.mu, dist.sigma];
    end
  end
  if saveDistributions
    saveBaselineKernelDists(baselineKernelDists, layerNumber);
  end

% --------------------------------------------------------------------
function newWeights = gennewWeightsFromDistributions1DGaussian( ...
  layers, ...
  layerNumber, ...
  baselineKernelDists)
% --------------------------------------------------------------------
  baselineWeights = layers{layerNumber}.weights{1};
  newWeights = [];
  for d = 1:size(baselineWeights, 4)
    for c = 1:size(baselineWeights, 3)
      baselineKernel = baselineWeights(:, :, c, d);
      matrixDims = [size(baselineKernel, 1), size(baselineKernel, 2)];
      vectorDims = [size(baselineKernel, 1) * size(baselineKernel, 2), 1];
      dist = makedist( ...
        'Normal', ...
        'mu', ...
        baselineKernelDists(1, 1, c, d), ...
        'sigma', ...
        baselineKernelDists(1, 2, c, d));
      vectorizedNewKernel = random(dist, vectorDims);
      newKernel = reshape(vectorizedNewKernel, matrixDims);
      newWeights(:, :, c, d) = newKernel;
    end
  end
  newWeights = single(newWeights);

% --------------------------------------------------------------------
function newWeights = saveBaselineKernelDists1DGaussian( ...
  baselineKernelDists, ...
  layerNumber)
% --------------------------------------------------------------------
  W1 = baselineKernelDists;
  save(sprintf('W1-baseline-kernel-dists-layer-%d.mat', layerNumber), 'W1');

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                                           -- ==
% -- ==                                          2D                                               -- ==
% -- ==                                                                                           -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

% --------------------------------------------------------------------
function newWeights = gen2DGaussianMultWeightsFromBaseline( ...
  layers, ...
  layerNumber)
% --------------------------------------------------------------------
  utils = gaussianUtils;
  newWeights = gen2DGaussianCoreWeightsFromBaseline( ...
    utils.fit2DGaussianAndDrawMultSamples, ...
    layers, ...
    layerNumber);

% --------------------------------------------------------------------
function newWeights = gen2DGaussianMult2WeightsFromBaseline( ...
  layers, ...
  layerNumber)
% --------------------------------------------------------------------
  utils = gaussianUtils;
  newWeights = gen2DGaussianCoreWeightsFromBaseline( ...
    utils.fit2DGaussianAndDrawMult2Samples, ...
    layers, ...
    layerNumber);

% --------------------------------------------------------------------
function newWeights = gen2DGaussianSuperWeightsFromBaseline( ...
  layers, ...
  layerNumber)
% --------------------------------------------------------------------
  utils = gaussianUtils;
  newWeights = gen2DGaussianCoreWeightsFromBaseline( ...
    utils.fit2DGaussianAndDrawSuperSamples, ...
    layers, ...
    layerNumber);

% --------------------------------------------------------------------
function newWeights = gen2DGaussianPositiveWeightsFromBaseline( ...
  layers, ...
  layerNumber)
% --------------------------------------------------------------------
  utils = gaussianUtils;
  newWeights = gen2DGaussianCoreWeightsFromBaseline( ...
    utils.fit2DGaussianAndDrawPositiveSamples, ...
    layers, ...
    layerNumber);

% --------------------------------------------------------------------
function newWeights = gen2DGaussianPosNegWeightsFromBaseline( ...
  layers, ...
  layerNumber)
% --------------------------------------------------------------------
  utils = gaussianUtils;
  newWeights = gen2DGaussianCoreWeightsFromBaseline( ...
    utils.fit2DGaussianAndDrawPosNegSamples, ...
    layers, ...
    layerNumber);

% --------------------------------------------------------------------
function newWeights = gen2DGaussianShiftFlipWeightsFromBaseline( ...
  layers, ...
  layerNumber)
% --------------------------------------------------------------------
  utils = gaussianUtils;
  newWeights = gen2DGaussianCoreWeightsFromBaseline( ...
    utils.fit2DGaussianAndDrawShiftFlipSamples, ...
    layers, ...
    layerNumber);

% --------------------------------------------------------------------
function newWeights = gen2DGaussianCoreWeightsFromBaseline( ...
  gen2DWeightsMethod, ...
  layers, ...
  layerNumber)
  % extract ~400,000 kernels, fit dist to each, and generate new kernel for each
% --------------------------------------------------------------------
  tic;
  utils = gaussianUtils;
  baselineWeights = layers{layerNumber}.weights{1};
  newWeights_W1 = [];
  for d = 1:size(baselineWeights, 4)
    for c = 1:size(baselineWeights, 3)
      baselineKernel = baselineWeights(:, :, c, d);
      newWeights_W1(:, :, c, d) = gen2DWeightsMethod(baselineKernel, false);
    end
  end
  newWeights{1} = single(newWeights_W1);
  % Don't use a dist, just put the baseline pretrain
  newWeights{2} = layers{layerNumber}.weights{2};
  numberOfKernels = size(newWeights{1}, 3) * size(newWeights{1}, 4);
  fprintf( ...
    ['[INFO] layer %d: 2D weights generated for %d kernels. ', ...
    'Elapsed Time (since start): %f\n'], ...
    layerNumber, ...
    numberOfKernels, ...
    toc);

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                                           -- ==
% -- ==                                         UTILS                                             -- ==
% -- ==                                                                                           -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

% --------------------------------------------------------------------
function newWeights = saveNewWeights(networkArch, weightInitType, newWeights, layerNumber)
  % WARNING: only 1 sample!
% --------------------------------------------------------------------
  % devPath = getDevPath();
  % folder = fullfile( ...
  %   devPath, ...
  %   'data', ...
  %   sprintf('cifar-%s', networkArch), ...
  %   sprintf('w_%s', weightInitType));
  % if ~exist(folder)
  %   mkdir(folder);
  % else
  %   % create backup of previous weights folder, then create new folder
  %   backup_folder = strcat(folder, sprintf('_bu_%s', datetime('now', 'Format', 'd-MMM-y-HH-mm-ss')));
  %   movefile(folder, backup_folder)
  %   mkdir(folder);
  % end
  % % then save the weights in the folder
  % W1 = newWeights{1};
  % W2 = newWeights{2};
  % save(fullfile(folder, sprintf('W1-layer-%d.mat', layerNumber)), 'W1');
  % save(fullfile(folder, sprintf('W2-layer-%d.mat', layerNumber)), 'W2');
  devPath = getDevPath();
  folder = fullfile( ...
    devPath, ...
    'data', ...
    sprintf('cifar-%s', networkArch), ...
    sprintf('w_%s', weightInitType));
  if ~exist(folder)
    mkdir(folder);
  end
  % then save the weights in the folder
  W1 = newWeights{1};
  W2 = newWeights{2};
  save(fullfile(folder, sprintf('W1-layer-%d.mat', layerNumber)), 'W1');
  save(fullfile(folder, sprintf('W2-layer-%d.mat', layerNumber)), 'W2');

% --------------------------------------------------------------------
function printNetworkStructure(net)
% --------------------------------------------------------------------
  layers = net.layers;
  for i = 1:numel(layers)
    if (strcmp(layers{i}.type, 'conv'))
      fprintf('-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- \n\n');
      fprintf('Layer %d', i);
      disp(layers{i}.weights);
    elseif (strcmp(layers{i}.type, 'relu'))
      fprintf('\trelu\n');
    elseif (strcmp(layers{i}.type, 'pool'))
      fprintf('\t');
      fprintf( ...
        'pool - method: %s, stride: %d, pool: ', ...
        layers{i}.method, ...
        layers{i}.stride);
      disp(layers{i}.pool);
    end
  end

% --------------------------------------------------------------------
function runInTryCatch(function_handle, networkArch, weightInitType)
% --------------------------------------------------------------------
  try
    feval(function_handle, networkArch, weightInitType);
  catch
    fprintf('caught ya bitch!\n');
  end
