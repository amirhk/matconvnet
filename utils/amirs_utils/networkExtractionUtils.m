function fh = networkExtractionUtils()
  % assign function handles so we can call these local functions from elsewhere
  fh.extractAlexNetCifar = @extractAlexNetCifar;
  fh.genRandomWeights = @genRandomWeights;
  fh.genRandomWeightsFromBaseline1DGaussian = @genRandomWeightsFromBaseline1DGaussian;
  fh.genRandomWeightsFromBaseline2DGaussian = @genRandomWeightsFromBaseline2DGaussian;
  fh.genRandomWeightsFromBaseline2DGaussianSuper = @genRandomWeightsFromBaseline2DGaussianSuper;

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
  % genWeightsMethod = @genRandomWeightsFromBaseline1DGaussian;
  genWeightsMethod = @genRandomWeightsFromBaseline2DGaussian;
  % genWeightsMethod = @genRandomWeightsFromBaseline2DGaussianSuper;
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
      randomWeights = genWeightsMethod(layers, layerNumber);
      saveRandomWeights(randomWeights, layerNumber);
    end
  end
  fprintf('[INFO] Successfully finished generating weights!\n\n');

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

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

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

% --------------------------------------------------------------------
function randomWeights = genRandomWeightsFromBaseline1DGaussian( ...
  layers, ...
  layerNumber)
  % extract ~400,000 kernels, save distributions, as well as random kernels from
  % those distributions
% --------------------------------------------------------------------
  tic;
  saveDistributions = false;
  baselineKernelDists = getDistsFromBaseline1DGaussian( ...
    layers, ...
    layerNumber, ...
    saveDistributions);
  randomWeights{1} = genRandomWeightsFromDistributions1DGaussian( ...
    layers, ...
    layerNumber, ...
    baselineKernelDists);
  % Don't use a dist, just put the baseline pretrain
  randomWeights{2} = layers{layerNumber}.weights{2};
  numberOfKernels = size(randomWeights{1}, 3) * size(randomWeights{1}, 4);
  fprintf( ...
    ['[INFO] layer %d: random weights generated for %d kernels. ', ...
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
function randomWeights = genRandomWeightsFromDistributions1DGaussian( ...
  layers, ...
  layerNumber, ...
  baselineKernelDists)
% --------------------------------------------------------------------
  baselineWeights = layers{layerNumber}.weights{1};
  randomWeights = [];
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
      vectorizedRandomKernel = random(dist, vectorDims);
      randomKernel = reshape(vectorizedRandomKernel, matrixDims);
      randomWeights(:, :, c, d) = randomKernel;
    end
  end
  randomWeights = single(randomWeights);

% --------------------------------------------------------------------
function randomWeights = saveBaselineKernelDists1DGaussian( ...
  baselineKernelDists, ...
  layerNumber)
% --------------------------------------------------------------------
  W1 = baselineKernelDists;
  save(sprintf('W1-baseline-kernel-dists-layer-%d.mat', layerNumber), 'W1');

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

% --------------------------------------------------------------------
function randomWeights = genRandomWeightsFromBaseline2DGaussian( ...
  layers, ...
  layerNumber)
  % extract ~400,000 kernels, save distributions, as well as random kernels from
  % those distributions
% --------------------------------------------------------------------
  tic;
  utils = gaussianUtils;
  baselineWeights = layers{layerNumber}.weights{1};
  randomWeights_W1 = [];
  for d = 1:size(baselineWeights, 4)
    for c = 1:size(baselineWeights, 3)
      baselineKernel = baselineWeights(:, :, c, d);
      randomWeights_W1(:, :, c, d) = ...
        utils.fit2DGaussianAndDrawSamples(baselineKernel, false);
    end
  end
  randomWeights{1} = single(randomWeights_W1);
  % Don't use a dist, just put the baseline pretrain
  randomWeights{2} = layers{layerNumber}.weights{2};
  numberOfKernels = size(randomWeights{1}, 3) * size(randomWeights{1}, 4);
  fprintf( ...
    ['[INFO] layer %d: random weights generated for %d kernels. ', ...
    'Elapsed Time (since start): %f\n'], ...
    layerNumber, ...
    numberOfKernels, ...
    toc);

% --------------------------------------------------------------------
function randomWeights = genRandomWeightsFromBaseline2DGaussianSuper( ...
  layers, ...
  layerNumber)
  % extract ~400,000 kernels, save distributions, as well as random kernels from
  % those distributions
% --------------------------------------------------------------------
  tic;
  utils = gaussianUtils;
  baselineWeights = layers{layerNumber}.weights{1};
  randomWeights_W1 = [];
  for d = 1:size(baselineWeights, 4)
    for c = 1:size(baselineWeights, 3)
      baselineKernel = baselineWeights(:, :, c, d);
      randomWeights_W1(:, :, c, d) = ...
        utils.fit2DGaussianAndDrawSuperSamples(baselineKernel, false);
    end
  end
  randomWeights{1} = single(randomWeights_W1);
  % Don't use a dist, just put the baseline pretrain
  randomWeights{2} = layers{layerNumber}.weights{2};
  numberOfKernels = size(randomWeights{1}, 3) * size(randomWeights{1}, 4);
  fprintf( ...
    ['[INFO] layer %d: random weights generated for %d kernels. ', ...
    'Elapsed Time (since start): %f\n'], ...
    layerNumber, ...
    numberOfKernels, ...
    toc);

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

% --------------------------------------------------------------------
function randomWeights = saveRandomWeights(randomWeights, layerNumber)
  % WARNING: only 1 sample!
% --------------------------------------------------------------------
  W1 = randomWeights{1};
  W2 = randomWeights{2};
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
