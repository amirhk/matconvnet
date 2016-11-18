function fh = networkExtractionUtils()
  % assign function handles so we can call these local functions from elsewhere
  fh.extractNewWeightsFromNetwork = @extractNewWeightsFromNetwork;
  fh.extractAllNewWeightsFromNetwork = @extractAllNewWeightsFromNetwork;

% --------------------------------------------------------------------
function extractNewWeightsFromNetwork(dataset, networkArch, weightInitType)
  % networkArch = {'alexnet', 'lenet', 'mnistnet'}
  % weightInitType = {'baseline', 'compRand', '1D', '2D-positive', '2D-super', '2D-posneg', '2D-shiftflip', '2D-mult-randn', '2D-mult-kernel'};
% --------------------------------------------------------------------
  fprintf('\n-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --\n');
  fprintf(sprintf('[INFO] Loading data from pre-trained %s on %s...\n', networkArch, dataset));
  devPath = getDevPath();
  loadedFile = load(fullfile( ...
    devPath, ...
    'data', ...
    sprintf('%s-%s', dataset, networkArch), ...
    'pretrained.mat'));
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
    case 'layerwise-1D'
      genWeightsMethod = @genLayerwise1DGaussianWeightsFromBaseline;
    case '2D-super'
      genWeightsMethod = @gen2DGaussianSuperWeightsFromBaseline;
    case '2D-posneg'
      genWeightsMethod = @gen2DGaussianPosNegWeightsFromBaseline;
    case '2D-positive'
      genWeightsMethod = @gen2DGaussianPositiveWeightsFromBaseline;
    case '2D-shiftflip'
      genWeightsMethod = @gen2DGaussianShiftFlipWeightsFromBaseline;
    case '2D-mult-randn'
      genWeightsMethod = @gen2DGaussianMultRandnWeightsFromBaseline;
    case '2D-mult-kernel'
      genWeightsMethod = @gen2DGaussianMultKernelWeightsFromBaseline;
  end
  genNewWeights(dataset, networkArch, weightInitType, net, genWeightsMethod);

% --------------------------------------------------------------------
function extractAllNewWeightsFromNetwork(dataset, networkArch)
  % networkArch = {'alexnet', 'lenet', 'mnistnet'}
% --------------------------------------------------------------------
  runInTryCatch(@extractNewWeightsFromNetwork, dataset, networkArch, 'baseline');
  runInTryCatch(@extractNewWeightsFromNetwork, dataset, networkArch, 'compRand');
  runInTryCatch(@extractNewWeightsFromNetwork, dataset, networkArch, '1D');
  runInTryCatch(@extractNewWeightsFromNetwork, dataset, networkArch, 'layerwise-1D');
  runInTryCatch(@extractNewWeightsFromNetwork, dataset, networkArch, '2D-super');
  runInTryCatch(@extractNewWeightsFromNetwork, dataset, networkArch, '2D-posneg');
  runInTryCatch(@extractNewWeightsFromNetwork, dataset, networkArch, '2D-positive');
  runInTryCatch(@extractNewWeightsFromNetwork, dataset, networkArch, '2D-shiftflip');
  runInTryCatch(@extractNewWeightsFromNetwork, dataset, networkArch, '2D-mult-randn');
  runInTryCatch(@extractNewWeightsFromNetwork, dataset, networkArch, '2D-mult-kernel');

% --------------------------------------------------------------------
function genNewWeights(dataset, networkArch, weightInitType, net, genWeightsMethod)
  % for all 'conv' layers...
% --------------------------------------------------------------------
  fprintf(sprintf('[INFO] Generating new `%s` weights from `%s`... \n', weightInitType, networkArch));
  layers = net.layers;
  for i = 1:numel(layers)
    if (strcmp(layers{i}.type, 'conv'))
      layerNumber = i;
      newWeights = genWeightsMethod(dataset, networkArch, layers, layerNumber);
      saveNewWeights(dataset, networkArch, weightInitType, newWeights, layerNumber);
    end
  end
  fprintf('[INFO] Successfully finished generating weights!\n\n');

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                                           -- ==
% -- ==                                 RANDOM / BASELINE                                         -- ==
% -- ==                                                                                           -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

% --------------------------------------------------------------------
function randomWeights = genBaselineWeights(dataset, networkArch, layers, layerNumber)
  % just save the baseline in the form of weights, not the whole network
% --------------------------------------------------------------------
  randomWeights{1} = layers{layerNumber}.weights{1};
  randomWeights{2} = layers{layerNumber}.weights{2};

% --------------------------------------------------------------------
function randomWeights = genCompRandWeights(dataset, networkArch, layers, layerNumber)
  % extract ~400,000 kernels and just save that many random kernels of the same
  % size generated from randn methods
% --------------------------------------------------------------------
  W1 = layers{layerNumber}.weights{1};
  k = size(W1, 1); % also k = size(W1, 2);
  m = size(W1, 3);
  n = size(W1, 4);
  switch networkArch
    case 'mnistnet'
      init_multiplier = 1/100;
    case 'lenet'
      switch layerNumber
        case 1
          init_multiplier = 1/100;
        otherwise
          init_multiplier = 5/100;
      end
    case 'alexnet'
      init_multiplier = 5/1000;
  end
  randomWeights{1} = init_multiplier * randn(k, k, m, n, 'single');
  randomWeights{2} = zeros(1, n, 'single');

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                                           -- ==
% -- ==                                          1D                                               -- ==
% -- ==                                                                                           -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

% --------------------------------------------------------------------
function newWeights = gen1DGaussianWeightsFromBaseline( ...
  dataset, ...
  networkArch, ...
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
% -- ==                                    layerwise-1D                                           -- ==
% -- ==                                                                                           -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

% --------------------------------------------------------------------
function newWeights = genLayerwise1DGaussianWeightsFromBaseline( ...
  dataset, ...
  networkArch, ...
  layers, ...
  layerNumber)
% --------------------------------------------------------------------
  tic;
  % vectorize all the kernels together, get global 1D distribution, sample from
  % that, reshape back to 4D matrix, and set the new value to newWeights{1}. For
  % newWeights{2} just use the baseline pretrain weights.
  baselineWeights = layers{layerNumber}.weights{1};
  matrixDims = size(baselineWeights);
  vectorDims = [size(baselineWeights, 1) * size(baselineWeights, 2) * size(baselineWeights, 3) * size(baselineWeights, 4), 1];
  vectorizedAllBaselineKernels = reshape(baselineWeights, vectorDims);
  dist = fitdist(vectorizedAllBaselineKernels, 'Normal');
  vectorizedAllNewKernels = random(dist, vectorDims);
  newKernel = reshape(vectorizedAllNewKernels, matrixDims);
  newWeights{1} = newKernel;
  newWeights{2} = layers{layerNumber}.weights{2};
  numberOfKernels = size(newWeights{1}, 3) * size(newWeights{1}, 4);
  fprintf( ...
    ['[INFO] layer %d: GLOBAL 1D weights generated for %d kernels. ', ...
    'Elapsed Time (since start): %f\n'], ...
    layerNumber, ...
    numberOfKernels, ...
    toc);


% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                                           -- ==
% -- ==                                          2D                                               -- ==
% -- ==                                                                                           -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

% --------------------------------------------------------------------
function newWeights = gen2DGaussianSuperWeightsFromBaseline( ...
  dataset, ...
  networkArch, ...
  layers, ...
  layerNumber)
% --------------------------------------------------------------------
  utils = gaussianUtils;
  newWeights = gen2DGaussianCoreWeightsFromBaseline( ...
    utils.fit2DGaussianAndDrawSuperSamples, ...
    layers, ...
    layerNumber);

% --------------------------------------------------------------------
function newWeights = gen2DGaussianPosNegWeightsFromBaseline( ...
  dataset, ...
  networkArch, ...
  layers, ...
  layerNumber)
% --------------------------------------------------------------------
  utils = gaussianUtils;
  newWeights = gen2DGaussianCoreWeightsFromBaseline( ...
    utils.fit2DGaussianAndDrawPosNegSamples, ...
    layers, ...
    layerNumber);

% --------------------------------------------------------------------
function newWeights = gen2DGaussianPositiveWeightsFromBaseline( ...
  dataset, ...
  networkArch, ...
  layers, ...
  layerNumber)
% --------------------------------------------------------------------
  utils = gaussianUtils;
  newWeights = gen2DGaussianCoreWeightsFromBaseline( ...
    utils.fit2DGaussianAndDrawPositiveSamples, ...
    layers, ...
    layerNumber);

% --------------------------------------------------------------------
function newWeights = gen2DGaussianShiftFlipWeightsFromBaseline( ...
  dataset, ...
  networkArch, ...
  layers, ...
  layerNumber)
% --------------------------------------------------------------------
  utils = gaussianUtils;
  newWeights = gen2DGaussianCoreWeightsFromBaseline( ...
    utils.fit2DGaussianAndDrawShiftFlipSamples, ...
    layers, ...
    layerNumber);

% --------------------------------------------------------------------
function newWeights = gen2DGaussianMultRandnWeightsFromBaseline( ...
  dataset, ...
  networkArch, ...
  layers, ...
  layerNumber)
% --------------------------------------------------------------------
  utils = gaussianUtils;
  newWeights = gen2DGaussianCoreWeightsFromBaseline( ...
    utils.fit2DGaussianAndDrawMultRandnSamples, ...
    layers, ...
    layerNumber);

% --------------------------------------------------------------------
function newWeights = gen2DGaussianMultKernelWeightsFromBaseline( ...
  dataset, ...
  networkArch, ...
  layers, ...
  layerNumber)
% --------------------------------------------------------------------
  utils = gaussianUtils;
  newWeights = gen2DGaussianCoreWeightsFromBaseline( ...
    utils.fit2DGaussianAndDrawMultKernelSamples, ...
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
function newWeights = saveNewWeights(dataset, networkArch, weightInitType, newWeights, layerNumber)
  % WARNING: only 1 sample!
% --------------------------------------------------------------------
  % devPath = getDevPath();
  % folder = fullfile( ...
  %   devPath, ...
  %   'data', ...
  %   sprintf('%s-%s', dataset, networkArch), ...
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
    sprintf('%s-%s', dataset, networkArch), ...
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
function runInTryCatch(function_handle, dataset, networkArch, weightInitType)
% --------------------------------------------------------------------
  % feval(function_handle, dataset, networkArch, weightInitType);
  try
    feval(function_handle, dataset, networkArch, weightInitType);
  catch
    fprintf('caught ya bitch!\n');
  end
