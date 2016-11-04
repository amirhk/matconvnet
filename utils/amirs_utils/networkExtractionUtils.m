function fh = networkExtractionUtils()
  % assign function handles so we can call these local functions from elsewhere
  fh.extractAlexNetCifar = @extractAlexNetCifar;
  fh.genRandomWeights = @genRandomWeights;
  fh.gen1DGaussianWeightsFromBaseline = @gen1DGaussianWeightsFromBaseline;
  fh.gen2DGaussianMultWeightsFromBaseline = @gen2DGaussianMultWeightsFromBaseline;
  fh.gen2DGaussianSuperWeightsFromBaseline = @gen2DGaussianSuperWeightsFromBaseline;
  fh.gen2DGaussianPosNegWeightsFromBaseline = @gen2DGaussianPosNegWeightsFromBaseline;
  fh.gen2DGaussianPositiveWeightsFromBaseline = @gen2DGaussianPositiveWeightsFromBaseline;

% --------------------------------------------------------------------
function extractAlexNetCifar()
% --------------------------------------------------------------------
  fprintf('[INFO] Loading data from pre-trained AlexNet on Cifar...\n');
  devPath = getDevPath();
  loadedFile = ...
    load(fullfile(devPath, 'data', 'cifar-alexnet', 'alexnet+8epoch.mat'));
  fprintf('[INFO] Loading data successful!\n\n');
  net = loadedFile.net;
  % printNetworkStructure(net);
  % TODO: flip these if need be!
  % genWeightsMethod = @genRandomWeights;
  % genWeightsMethod = @gen1DGaussianWeightsFromBaseline;
  genWeightsMethod = @gen2DGaussianMultWeightsFromBaseline;
  % genWeightsMethod = @gen2DGaussianSuperWeightsFromBaseline;
  % genWeightsMethod = @gen2DGaussianPosNegWeightsFromBaseline;
  % genWeightsMethod = @gen2DGaussianPositiveWeightsFromBaseline;
  genNewWeights(net, genWeightsMethod);

% --------------------------------------------------------------------
function genNewWeights(net, genWeightsMethod)
  % for all 'conv' layers...
% --------------------------------------------------------------------
  fprintf('[INFO] Generating new weights... \n');
  layers = net.layers;
  for i = 1:numel(layers)
    if (strcmp(layers{i}.type, 'conv'))
      layerNumber = i;
      newWeights = genWeightsMethod(layers, layerNumber);
      saveNewWeights(newWeights, layerNumber);
    end
  end
  fprintf('[INFO] Successfully finished generating weights!\n\n');

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                                           -- ==
% -- ==                                        RANDOM                                             -- ==
% -- ==                                                                                           -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

% --------------------------------------------------------------------
function randomWeights = genRandomWeights(layers, layerNumber)
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
    ['[INFO] layer %d: 2D weights gene1rated for %d kernels. ', ...
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
function newWeights = saveNewWeights(newWeights, layerNumber)
  % WARNING: only 1 sample!
% --------------------------------------------------------------------
  W1 = newWeights{1};
  W2 = newWeights{2};
  save(sprintf('W1-layer-%d.mat', layerNumber), 'W1');
  save(sprintf('W2-layer-%d.mat', layerNumber), 'W2');

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
function datapath = getDevPath()
% --------------------------------------------------------------------
  if ispc
    datapath = 'H:\Amir';
  else
    datapath = '/Users/a6karimi/dev';
  end
