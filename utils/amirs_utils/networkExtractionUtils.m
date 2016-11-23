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
    'generated_weights', ...
    sprintf('%s', networkArch), ...
    sprintf('pretrained-%s-%s.mat', dataset, networkArch)));
  fprintf('[INFO] Loading data successful!\n\n');
  net = loadedFile.net;
  % printNetworkStructure(net);
  % TODO: change these....
  switch weightInitType
    case 'baseline'
      genWeightsMethod = @genBaselineWeights;
    case 'compRand'
      genWeightsMethod = @genCompRandWeights;
    case 'kernelwise-1D'
      genWeightsMethod = @genKernelwise1DGaussianWeightsFromBaseline;
    case 'layerwise-1D'
      genWeightsMethod = @genLayerwise1DGaussianWeightsFromBaseline;
    case 'clustered-layerwise-1D'
      genWeightsMethod = @genClusteredLayerwise1DGaussianWeightsFromBaseline;
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
  % runInTryCatch(@extractNewWeightsFromNetwork, dataset, networkArch, 'baseline');
  % runInTryCatch(@extractNewWeightsFromNetwork, dataset, networkArch, 'compRand');
  % runInTryCatch(@extractNewWeightsFromNetwork, dataset, networkArch, 'kernelwise-1D');
  % runInTryCatch(@extractNewWeightsFromNetwork, dataset, networkArch, 'layerwise-1D');
  runInTryCatch(@extractNewWeightsFromNetwork, dataset, networkArch, 'clustered-layerwise-1D');
  % runInTryCatch(@extractNewWeightsFromNetwork, dataset, networkArch, '2D-super');
  % runInTryCatch(@extractNewWeightsFromNetwork, dataset, networkArch, '2D-posneg');
  % runInTryCatch(@extractNewWeightsFromNetwork, dataset, networkArch, '2D-positive');
  % runInTryCatch(@extractNewWeightsFromNetwork, dataset, networkArch, '2D-shiftflip');
  % runInTryCatch(@extractNewWeightsFromNetwork, dataset, networkArch, '2D-mult-randn');
  % runInTryCatch(@extractNewWeightsFromNetwork, dataset, networkArch, '2D-mult-kernel');

% --------------------------------------------------------------------
function genNewWeights(dataset, networkArch, weightInitType, net, genWeightsMethod)
  % for all 'conv' layers...
% --------------------------------------------------------------------
  fprintf(sprintf('[INFO] Generating new `%s` weights from `%s`... \n\n', weightInitType, networkArch));
  layers = net.layers;
  for i = 1:numel(layers)
    if (strcmp(layers{i}.type, 'conv'))
      layerNumber = i;
      newWeights = genWeightsMethod(dataset, networkArch, layers, layerNumber);
      saveNewWeights(dataset, networkArch, weightInitType, newWeights, layerNumber);
    end
  end
  fprintf('[INFO] Successfully finished generating weights!\n');

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
function newWeights = genKernelwise1DGaussianWeightsFromBaseline( ...
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
  newWeights{1} = genNewWeightsFromDistributions1DGaussian( ...
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
function newWeights = genNewWeightsFromDistributions1DGaussian( ...
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
  baseline_weights = layers{layerNumber}.weights{1};
  % genNewKernelsFromApplying1DFitToBaselineKernels
  utils = gaussianUtils;
  matrix_dims = size(baseline_weights);
  vector_dims = [size(baseline_weights, 1) * size(baseline_weights, 2) * size(baseline_weights, 3) * size(baseline_weights, 4), 1];
  baseline_vectorized_kernels = reshape(baseline_weights, vector_dims);
  new_vectorized_kernels = utils.fit1DGaussianAndDrawSamples(baseline_vectorized_kernels);
  new_kernels = reshape(new_vectorized_kernels, matrix_dims);
  % set new_kernels
  newWeights{1} = single(new_kernels);
  newWeights{2} = layers{layerNumber}.weights{2};
  numberOfKernels = size(newWeights{1}, 3) * size(newWeights{1}, 4);
  fprintf( ...
    ['[INFO] layer %d: Layerwise 1D weights generated for %d kernels. ', ...
    'Elapsed Time (since start): %f\n'], ...
    layerNumber, ...
    numberOfKernels, ...
    toc);

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                                           -- ==
% -- ==                               clustered-layerwise-1D                                      -- ==
% -- ==                                                                                           -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

% --------------------------------------------------------------------
function newWeights = genClusteredLayerwise1DGaussianWeightsFromBaseline( ...
  dataset, ...
  networkArch, ...
  layers, ...
  layerNumber)
% --------------------------------------------------------------------
  tic;
  % 1. get baseline_kernels
  baseline_kernels = layers{layerNumber}.weights{1};

  % 2. convert baseline_kernels to size-format for k-means
  ydim = size(baseline_kernels, 1);
  xdim = size(baseline_kernels, 2);
  zdim = size(baseline_kernels, 3);
  tdim = size(baseline_kernels, 4);
  assert(ydim == xdim);
  data = reshape(baseline_kernels, ydim * xdim, zdim * tdim);
  data = data';
  % NOTE: reshape(A, a, b)' is NOT equal to reshape(A, b, a).

  % 3. divide data into k clusters
  cluster_count = 4;
  clusters = getDataInKClusters(data, cluster_count);

  % 4. for each cluster, find the distribution
  utils = gaussianUtils;
  for i = 1:cluster_count
    tmp = reshape(clusters{i}.data, [], 1);
    clusters{i}.dist = fitdist(tmp, 'Normal');
    clusters{i}.num_kernels_in_cluster = size(clusters{i}.data, 1);
  end

  % 5. for each kernel index, find the dist of the associated cluster and sample
  %    a new kernel (ydim * xdim elements)
  new_kernels = zeros(ydim, xdim, zdim, tdim);
  for i = 1:cluster_count
    for j = 1:length([clusters{i}.indices_in_data_matrix])
      index = clusters{i}.indices_in_data_matrix(j);
      z_index_in_4D_matrix = mod(index, zdim); if z_index_in_4D_matrix == 0; z_index_in_4D_matrix = zdim; end;
      t_index_in_4D_matrix = (index - z_index_in_4D_matrix) / zdim + 1;
      % fprintf('index: %d, z: %d, t:%d\n', index, z_index_in_4D_matrix, t_index_in_4D_matrix);
      new_sampled_kernel_from_cluster_distribution = random(clusters{i}.dist, [ydim, xdim]);
      new_kernels(:,:,z_index_in_4D_matrix, t_index_in_4D_matrix) = new_sampled_kernel_from_cluster_distribution;
    end
  end

  % for i = 1:cluster_count;
  %   fprintf('[INFO] cluster %d: mu = %6.5f, sigma = %6.5f\n', i, clusters{i}.dist.mu, clusters{i}.dist.sigma);
  % end

  % 6. set new_kernels
  newWeights{1} = single(new_kernels);
  newWeights{2} = layers{layerNumber}.weights{2};
  numberOfKernels = size(newWeights{1}, 3) * size(newWeights{1}, 4);
  fprintf( ...
    ['[INFO] layer %d: Clustered Layerwise 1D weights generated for %d kernels. ', ...
    'Elapsed Time (since start): %f\n'], ...
    layerNumber, ...
    numberOfKernels, ...
    toc);

% --------------------------------------------------------------------
function clusters = getDataInKClusters(data, k)
  % data is of size [n, k] where n is the number of vectors, k the dim of each
% --------------------------------------------------------------------
  idx = kmeans(data, k, 'maxIter', 250);
  clustered_data = {};
  for i = 1:k
    clusters{i}.data = data(find(idx == i), :);
    clusters{i}.indices_in_data_matrix = find(idx == i);
  end

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

  if strcmp(weightInitType, 'compRand')
    subDirName = 'w-compRand';
  else
    subDirName = sprintf('w-%s-from-%s', weightInitType, dataset);
  end

  directory = fullfile( ...
    devPath, ...
    'data', ...
    'generated_weights', ...
    sprintf('%s', networkArch), ...
    subDirName);
  if ~exist(directory)
    mkdir(directory);
  end
  % then save the weights in the directory
  W1 = newWeights{1};
  W2 = newWeights{2};
  save(fullfile(directory, sprintf('W1-layer-%d.mat', layerNumber)), 'W1');
  save(fullfile(directory, sprintf('W2-layer-%d.mat', layerNumber)), 'W2');

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
