% -------------------------------------------------------------------------
function fh = projectionUtils()
% -------------------------------------------------------------------------
% Copyright (c) 2017, Amir-Hossein Karimi
% All rights reserved.

% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution

% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.

  fh.getAngleSeparatedImdb = @getAngleSeparatedImdb;
  fh.getDenslyProjectedImdb = @getDenslyProjectedImdb;
  fh.getDenslyDownProjectedImdb = @getDenslyDownProjectedImdb;
  fh.getDenslyProjectedAndNormalizedImdb = @getDenslyProjectedAndNormalizedImdb;
  fh.getDenslyLogNormalProjectedImdb = @getDenslyLogNormalProjectedImdb;
  fh.getSparselyProjectedImdb = @getSparselyProjectedImdb;
  fh.projectImdbThroughNetwork = @projectImdbThroughNetwork;
  fh.getProjectionNetworkObject = @getProjectionNetworkObject;


% -------------------------------------------------------------------------
function projected_imdb = projectImdbThroughNetwork(imdb, net, forward_pass_depth)
% -------------------------------------------------------------------------
  if numel(net.layers) == 0 % net = 'larpV0P0RL0'
    projected_imdb = imdb;
    return
  end

  % get imdb
  input_imdb = imdb;
  train_imdb = filterImdbForSet(input_imdb, 1, 3);
  test_imdb = filterImdbForSet(input_imdb, 3, 3);

  % get net
  if forward_pass_depth == -1
    forward_pass_depth = numel(net.layers); % +1 necessary?.... guess not!
  end

  % get resulting matrix from forward pass for all samples
  all_train_samples_forward_pass_results = getProjectedImdbSamplesOnNet(train_imdb, net, forward_pass_depth);
  all_test_samples_forward_pass_results = getProjectedImdbSamplesOnNet(test_imdb, net, forward_pass_depth);

  % put it all together
  data_train = all_train_samples_forward_pass_results;
  data_test = all_test_samples_forward_pass_results;
  labels_train = train_imdb.images.labels;
  labels_test = test_imdb.images.labels;
  data = single(cat(4, data_train, data_test));
  labels = single(cat(2, labels_train, labels_test));
  set = single(cat(2, 1 * ones(1, size(labels_train, 2)), 3 * ones(1, size(labels_test, 2))));

  % shuffle
  % ix = randperm(size(data, 4));
  ix = 1:size(data, 4); %  TODO: this should be randomized!?!?!?!?!?!??!
  data = data(:,:,:,ix);
  labels = labels(ix);
  set = set(ix);

  % put it all together
  projected_imdb.images.data = data;
  projected_imdb.images.labels = labels;
  projected_imdb.images.set = set;
  projected_imdb.meta.sets = {'train', 'val', 'test'};

  % sanity
  assert(numel(input_imdb.images.set) == numel(projected_imdb.images.set))

% -------------------------------------------------------------------------
function net = getProjectionNetworkObject(dataset, larp_network_arch, larp_weight_init_sequence)
% -------------------------------------------------------------------------
  opts.general.dataset          = dataset;
  opts.general.network_arch     = larp_network_arch;
  opts.train.learning_rate      = [999*ones(1,1)]; % doesn't matter, as we're not training....
  opts.net.weight_init_source   = 'gen';
  opts.net.weight_init_sequence = larp_weight_init_sequence;

  network_opts = cnnInit(opts);
  net = network_opts.net;

% -------------------------------------------------------------------------
function projected_samples = getProjectedImdbSamplesOnNet(imdb, net, depth)
% -------------------------------------------------------------------------
  assert(numel(find(imdb.images.set == 1)) == 0); % cnn_train only projects test data. sad, i know.
  [net, info] = cnnTrain(net, imdb, getBatch(), ...
    'gpus', ifNotMacSetGpu(2), ...
    'forward_pass_only_mode', true, ...
    'forward_pass_only_depth', depth + 1, ... % +1 is critical because for a 3 layer network, cnn_train's res variable has 4 layers incl'd the input.
    'debug_flag', false, ...
    'continue', false, ...
    'num_epochs', 1, ...
    'train', [], ...
    'val', find(imdb.images.set == 3));
  projected_samples = info.all_samples_forward_pass_results;





% -------------------------------------------------------------------------
function fn = getBatch()
% -------------------------------------------------------------------------
  fn = @(x,y) getSimpleNNBatch(x,y);

% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
  images = imdb.images.data(:,:,:,batch);
  labels = imdb.images.labels(1,batch);
  if rand > 0.5, images=fliplr(images); end









% -------------------------------------------------------------------------
function imdb = getDenslyDownProjectedImdb(imdb, number_of_projection_layers, projection_layer_type, number_of_non_linear_layers, non_linear_layer_type, projected_dim)
% -------------------------------------------------------------------------

  %       IMPORTANT: REVERT TO ME BECAUSE CODE BELOW HAS WIERD ASSUMPTIONS
  % assert(number_of_projection_layers == 1);
  % assert(number_of_projection_layers <= 1);
  % assert(number_of_non_linear_layers <= number_of_projection_layers);

  % vectorized_imdb = getVectorizedImdb(imdb);
  % original_dim = size(vectorized_imdb.images.data, 2);
  % % projected_dim = original_dim;
  % non_linear_layer_count = 0;
  % for i = 1 : number_of_projection_layers
  %   random_projection_matrix = randn(projected_dim, original_dim);
  %   imdb = projectImdbUsingMatrix(imdb, random_projection_matrix);
  %   if non_linear_layer_count < number_of_non_linear_layers
  %     switch non_linear_layer_type
  %       case 'relu'
  %         imdb.images.data(imdb.images.data < 0) = 0;
  %       case 'sigmoid'
  %         imdb.images.data = logsig(imdb.images.data);
  %       case 'tanh'
  %         imdb.images.data = tanh(imdb.images.data);
  %     end
  %     non_linear_layer_count = non_linear_layer_count + 1;
  %   end
  % end




  % assert(number_of_projection_layers == 1);
  % assert(number_of_projection_layers <= 1);
  assert(number_of_non_linear_layers <= number_of_projection_layers);

  vectorized_imdb = getVectorizedImdb(imdb);
  original_dim = size(vectorized_imdb.images.data, 2);
  % projected_dim = original_dim;
  non_linear_layer_count = 0;
  for i = 1 : number_of_projection_layers
    if i == 1
      tmp_dim = original_dim; % this is original_dim if there's only 1 layer, or an evolving dim as we project further and further
    else
      tmp_dim = projected_dim; % this is original_dim if there's only 1 layer, or an evolving dim as we project further and further
    end

    switch projection_layer_type
      case 'dense_gaussian'
        % keyboard
        random_projection_matrix = randn(projected_dim, tmp_dim) / sqrt(projected_dim);
      case 'dense_log_normal'
        lognormal_mean = -0.702;
        lognormal_var = 0.9355;
        mu = log( (lognormal_mean ^ 2) / sqrt(lognormal_var + lognormal_mean ^ 2));
        sigma = sqrt( log( lognormal_var / (lognormal_mean ^ 2) + 1));
        vectorized_generated_samples = lognrnd(mu, sigma, 1, (original_dim)^2);
        random_projection_matrix = reshape(generated_samples, projected_dim, tmp_dim);
      end
    % result below is 4D imdb
    imdb = projectImdbUsingMatrix(imdb, random_projection_matrix);
    if non_linear_layer_count < number_of_non_linear_layers
      switch non_linear_layer_type
        case 'relu'
          imdb.images.data(imdb.images.data < 0) = 0;
        case 'pooling-max-3x3-stride-2-pad-0101'
          % great, we can apply vl_nnpool on top of a 4D data...
          imdb.images.data = vl_nnpool(single(imdb.images.data), [3, 3], 'Stride', 2, 'Pad', [0 1 0 1], 'method', 'max');
        case 'relu_w_pooling-max-3x3-stride-2-pad-0101'
          % great, we can apply vl_nnpool on top of a 4D data...
          imdb.images.data = vl_nnpool(single(imdb.images.data), [3, 3], 'Stride', 2, 'Pad', [0 1 0 1], 'method', 'max');
          imdb.images.data(imdb.images.data < 0) = 0;
        case 'sigmoid'
          imdb.images.data = logsig(imdb.images.data);
        case 'tanh'
          imdb.images.data = tanh(imdb.images.data);
        otherwise
          throwException('[ERROR] non_linear_layer_type not recognized.');
      end
      non_linear_layer_count = non_linear_layer_count + 1;
    end
  end

% -------------------------------------------------------------------------
function imdb = getDenslyProjectedImdb(imdb, number_of_projection_layers, number_of_relu_layers)
% -------------------------------------------------------------------------
  assert(number_of_relu_layers <= number_of_projection_layers);
  vectorized_imdb = getVectorizedImdb(imdb);
  original_dim = size(vectorized_imdb.images.data, 2);
  projected_dim = original_dim;
  relu_count = 0;
  for i = 1 : number_of_projection_layers
    % random_projection_matrix = randn(projected_dim, original_dim) * 1/100;
    random_projection_matrix = randn(projected_dim, original_dim);
    % random_projection_matrix = randn(projected_dim, original_dim) / sqrt(original_dim);
    % random_projection_matrix = randn(projected_dim, original_dim) / sqrt(sqrt(original_dim));
    % random_projection_matrix = randn(projected_dim, original_dim) / sqrt(original_dim) * sqrt(2);
    % random_projection_matrix = (round(rand(projected_dim, original_dim)) - 0.5) * 2 / sqrt(original_dim); % Bernoulli
    imdb = projectImdbUsingMatrix(imdb, random_projection_matrix);
    if relu_count < number_of_relu_layers
      % apply relu
      imdb.images.data(imdb.images.data < 0) = 0;
      relu_count = relu_count + 1;
    end
  end
  % random_projection_matrix = randn(projected_dim, original_dim) * 1/100;
  % imdb = projectImdbUsingMatrix(imdb, random_projection_matrix);

% -------------------------------------------------------------------------
function imdb = getDenslyProjectedAndNormalizedImdb(imdb, number_of_projection_layers, number_of_relu_layers)
% -------------------------------------------------------------------------
  assert(number_of_relu_layers <= number_of_projection_layers);
  vectorized_imdb = getVectorizedImdb(imdb);
  original_dim = size(vectorized_imdb.images.data, 2);
  projected_dim = original_dim;
  relu_count = 0;
  for i = 1 : number_of_projection_layers
    random_projection_matrix = randn(projected_dim, original_dim);
    [Q, R] = qr(random_projection_matrix);
    random_projection_matrix = normr(Q);
    imdb = projectImdbUsingMatrix(imdb, random_projection_matrix);
    if relu_count < number_of_relu_layers
      % apply relu
      imdb.images.data(imdb.images.data < 0) = 0;
      relu_count = relu_count + 1;
    end
  end

% -------------------------------------------------------------------------
function imdb = getDenslyLogNormalProjectedImdb(imdb, number_of_projection_layers, number_of_relu_layers)
% -------------------------------------------------------------------------
  assert(number_of_relu_layers <= number_of_projection_layers);
  vectorized_imdb = getVectorizedImdb(imdb);
  original_dim = size(vectorized_imdb.images.data, 2);
  projected_dim = original_dim;
  relu_count = 0;
  for i = 1 : number_of_projection_layers
    lognormal_mean = -0.702;
    lognormal_var = 0.9355;
    mu = log( (lognormal_mean ^ 2) / sqrt(lognormal_var + lognormal_mean ^ 2));
    sigma = sqrt( log( lognormal_var / (lognormal_mean ^ 2) + 1));
    generated_samples = lognrnd(mu, sigma, 1, (original_dim)^2);
    random_projection_matrix = reshape(generated_samples, projected_dim, original_dim);
    imdb = projectImdbUsingMatrix(imdb, random_projection_matrix);
    if relu_count < number_of_relu_layers
      % apply relu
      imdb.images.data(imdb.images.data < 0) = 0;
      relu_count = relu_count + 1;
    end
  end


% -------------------------------------------------------------------------
function imdb = getSparselyProjectedImdb(imdb, number_of_projection_layers, number_of_relu_layers)
% -------------------------------------------------------------------------
  % TODO....


% -------------------------------------------------------------------------
function imdb = projectImdbUsingMatrix(imdb, projection_matrix)
% -------------------------------------------------------------------------
  vectorized_original_imdb = getVectorizedImdb(imdb);

  original_dim = size(vectorized_original_imdb.images.data, 2);
  projected_dim = size(projection_matrix, 1);
  number_of_samples = size(vectorized_original_imdb.images.data, 1);

  all_data_original_vectorized = vectorized_original_imdb.images.data;
  all_data_projected_vectorized = all_data_original_vectorized * projection_matrix';

  vectorized_projected_imdb = vectorized_original_imdb;
  vectorized_projected_imdb.images.data = all_data_projected_vectorized;

  % imdb = get4DImdb(vectorized_projected_imdb, projected_dim, 1, 1, number_of_samples);
  imdb = get4DImdb(vectorized_projected_imdb, sqrt(projected_dim), sqrt(projected_dim), 1, number_of_samples);











% -------------------------------------------------------------------------
function imdb = getAngleSeparatedImdb(input_imdb)
% -------------------------------------------------------------------------
  % IMPORTANT!! S & D are computed for the training set only, but applied to
  % both training and test sets.
  [S, D] = getSimilarityAndDissimilarityEnumerationSets(input_imdb);
  printConsoleOutputSeparator();
  M_S = getCovarianceMeasureForSet(input_imdb, S, 'S');
  M_D = getCovarianceMeasureForSet(input_imdb, D, 'D');
  printConsoleOutputSeparator();

  % Computing right eigen vectors.
  % jigar = M_S;
  % M_S = M_D;
  % M_D = jigar;

  tmp = inv(M_D);
  if ~isinf(tmp) % && ~isnan(tmp)
    [V, ~] = eig(tmp * M_S);
    % [V, ~] = eig(pinv(M_D) * M_S);
    % [V, ~] = eig(M_S - M_D);
    angle_separation_matrix = V';
  else
    angle_separation_matrix = eye(size(M_D));
  end

  % keyboard
  angle_separation_matrix = real(angle_separation_matrix);
  % assert(isreal(angle_separation_matrix));

  imdb = projectImdbUsingMatrix(input_imdb, angle_separation_matrix);


% -------------------------------------------------------------------------
function [S, D] = getSimilarityAndDissimilarityEnumerationSets(imdb)
% -------------------------------------------------------------------------
  unique_labels = unique(imdb.images.labels);
  number_of_classes = numel(unique_labels);

  S = {};
  D = {};
  for i = 1:number_of_classes
    afprintf(sprintf('Populating `S` and `D` sets for class #%d\n', i));
    S = cat(2, S, getEnumerationsOfSampleIndicesFromClasses(imdb, i, i));
    for j = i+1:number_of_classes
      D = cat(2, D, getEnumerationsOfSampleIndicesFromClasses(imdb, i, j));
    end
  end


% -------------------------------------------------------------------------
function enumerations = getEnumerationsOfSampleIndicesFromClasses(imdb, class_1_label, class_2_label)
% -------------------------------------------------------------------------
  % class_1_indices = find(imdb.images.labels == class_1_label);
  % class_2_indices = find(imdb.images.labels == class_2_label);
  % training samples only!!!!!!!!!!!!
  class_1_indices = bsxfun(@and, imdb.images.labels == class_1_label, imdb.images.set == 1);
  class_1_indices = find(class_1_indices == 1);
  class_2_indices = bsxfun(@and, imdb.images.labels == class_2_label, imdb.images.set == 1);
  class_2_indices = find(class_2_indices == 1);
  if class_1_label == class_2_label
    assert(isequal(class_1_indices, class_2_indices));
    enumerations = getEnumerationsOfTwoSimilarVectors(class_1_indices, class_2_indices);
  else
    assert(sum(ismember(class_1_indices, class_2_indices)) == 0);
    enumerations = getEnumerationsOfTwoDissimilarVectors(class_1_indices, class_2_indices);
  end


% -------------------------------------------------------------------------
function enumerations = getEnumerationsOfTwoSimilarVectors(a, b)
% -------------------------------------------------------------------------
  enumerations = {};
  for i = 1:length(a)
    for j = i+1:length(b)
      enumerations{end+1} = [a(i), b(j)];
    end
  end


% -------------------------------------------------------------------------
function enumerations = getEnumerationsOfTwoDissimilarVectors(a, b)
% -------------------------------------------------------------------------
  [A, B] = meshgrid(a, b);
  c = cat(2, A', B');
  d = reshape(c, [], 2);
  enumerations = {};
  for i = 1:size(d, 1)
    enumerations{end+1} = d(i, :);
  end


% -------------------------------------------------------------------------
function M = getCovarianceMeasureForSet(imdb, input_set, input_set_name)
  % input_set_name = {'S', 'D'}
% -------------------------------------------------------------------------
  load_from_saved_measure_file_if_present = true;
  saved_measure_file =  fullfile( ...
    getDevPath(), ...
    'data', ...
    'similarity_dissimilarity_sets', ...
    sprintf('%s_measure_for_%s.mat', input_set_name, imdb.name));
  if exist(saved_measure_file) == 2 && load_from_saved_measure_file_if_present
    afprintf(sprintf('[INFO] loading saved %s measure...\n', input_set_name));
    tmp = load(saved_measure_file);
    M = tmp.M;
    afprintf(sprintf('done!\n'));
  else
    M = 0;
    afprintf(sprintf('[INFO] processing sample pairs # '));
    for k = 1:length(input_set)
      for j = 0:log10(k - 1) + (3 + numel(num2str(length(input_set))))
        fprintf('\b'); % delete previous counter display
      end
      fprintf('%d / %d', k, length(input_set));
      pair_of_samples_indices = input_set(k);
      sample_i = getVectorizedSampleAtIndex(imdb, pair_of_samples_indices{1}(1));
      sample_j = getVectorizedSampleAtIndex(imdb, pair_of_samples_indices{1}(2));
      M = M + getCovarianceThingyBetweenVectors(sample_i', sample_j');
    end
    afprintf(sprintf('\n'));
    M = (1 / length(input_set)) * M;

    save(saved_measure_file, 'M');
  end


% -------------------------------------------------------------------------
function matrix = getCovarianceThingyBetweenVectors(a, b)
% -------------------------------------------------------------------------
  matrix = (a - b) * (a - b)';



























