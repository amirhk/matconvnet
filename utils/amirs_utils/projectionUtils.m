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
  fh.getPCAProjectedImdb = @getPCAProjectedImdb;
  fh.getDenselyProjectedImdb = @getDenselyProjectedImdb;
  fh.getDenselyDownProjectedImdb = @getDenselyDownProjectedImdb;
  fh.getEnsembleDenselyDownProjectedImdb = @getEnsembleDenselyDownProjectedImdb;
  fh.getDenselyProjectedAndNormalizedImdb = @getDenselyProjectedAndNormalizedImdb;
  fh.getDenselyLogNormalProjectedImdb = @getDenselyLogNormalProjectedImdb;
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
    forward_pass_depth = numel(net.layers);
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
  ix = randperm(size(data, 4));
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
  assert(numel(find(imdb.images.set == 1)) == 0); % cnnTrain only projects test data. sad, i know.
  [net, info] = cnnTrain(net, imdb, getBatch(), ...
    'gpus', ifNotMacSetGpu(1), ...
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
  labels = imdb.images.labels(batch);
  % if rand > 0.5, images=fliplr(images); end









% -------------------------------------------------------------------------
function imdb = getPCAProjectedImdb(imdb, projected_dim)
% -------------------------------------------------------------------------

  original_train_imdb = filterImdbForSet(imdb, 1, 1);
  original_test_imdb = filterImdbForSet(imdb, 3, 3);


  vectorized_original_train_imdb = getVectorizedImdb(original_train_imdb);
  vectorized_original_test_imdb = getVectorizedImdb(original_test_imdb);

  assert(projected_dim <= size(vectorized_original_train_imdb.images.data, 2));

  number_of_train_samples = size(vectorized_original_train_imdb.images.data, 1);
  number_of_test_samples = size(vectorized_original_test_imdb.images.data, 1);


  % [coeff, score, latent, tsquared, explained, mu] = pca(vectorized_original_train_imdb.images.data);

  % % approximationRank1 = score(:,1) * coeff(:,1)' + repmat(mu, 100, 1);
  % data_train_approximation = score(:,1:projected_dim) * coeff(:,1:projected_dim)' + repmat(mu, number_of_train_samples, 1);
  % data_test_approximation = score(:,1:projected_dim) * coeff(:,1:projected_dim)' + repmat(mu, number_of_test_samples, 1);

  % keyboard

  % data_train_approximation = vectorized_original_train_imdb.images.data * coeff(:, 1 : projected_dim) + repmat(mu(1:projected_dim), number_of_train_samples, 1);
  % data_test_approximation = vectorized_original_test_imdb.images.data * coeff(:, 1 : projected_dim) + repmat(mu(1:projected_dim), number_of_test_samples, 1);



  X = vectorized_original_train_imdb.images.data;
  X = bsxfun(@minus, X, mean(X,1));           %# zero-center
  C = (X'*X)./(size(X,1)-1);                  %'# cov(X)

  [V D] = eig(C);
  [D order] = sort(diag(D), 'descend');       %# sort cols high to low
  V = V(:,order);

  % TODO: IMPORTANT: should add mean back after projection... not needed for 1-NN, but for accuracy!

  % newX = X*V(:,1:end);

  data_train_approximation = vectorized_original_train_imdb.images.data * V(:,1:projected_dim);
  data_test_approximation = vectorized_original_test_imdb.images.data * V(:,1:projected_dim);


  vectorized_projected_train_imdb = vectorized_original_train_imdb;
  vectorized_projected_test_imdb = vectorized_original_test_imdb;

  vectorized_projected_train_imdb.images.data = data_train_approximation;
  vectorized_projected_test_imdb.images.data = data_test_approximation;

  train_imdb = get4DImdb(vectorized_projected_train_imdb, projected_dim, 1, 1, number_of_train_samples);
  test_imdb = get4DImdb(vectorized_projected_test_imdb, projected_dim, 1, 1, number_of_test_samples);

  imdb = mergeImdbs(train_imdb, test_imdb);


% -------------------------------------------------------------------------
function imdb = getEnsembleDenselyDownProjectedImdb(imdb, number_of_projection_layers, projection_layer_type, number_of_non_linear_layers, non_linear_layer_type, projected_dim, number_in_ensemble)
% -------------------------------------------------------------------------
  tmp_imdbs = {};
  for i = 1 : number_in_ensemble
    tmp_imdbs{end+1} = getDenselyDownProjectedImdb(imdb, number_of_projection_layers, projection_layer_type, number_of_non_linear_layers, non_linear_layer_type, projected_dim);
  end

  final_merged_imdb = imdb;
  final_merged_imdb.images.data = zeros(size(tmp_imdbs{1}.images.data, 1), size(tmp_imdbs{1}.images.data, 2), number_in_ensemble, size(tmp_imdbs{1}.images.data, 4));
  for j = 1 : size(imdb.images.data, 4)
    % tmp_image = zeros(size(tmp_imdbs{1}.images.data, 1), size(tmp_imdbs{1}.images.data, 2), number_in_ensemble);
    for i = 1 : number_in_ensemble
      assert(size(tmp_imdbs{i}.images.data(:,:,:,j), 3) == 1); % the dense RP creates a 2D image from a 2D or 3D image
      final_merged_imdb.images.data(:,:,i,j) = tmp_imdbs{i}.images.data(:,:,:,j);
    end
    % final_merged_imdb.images.data(:,:,:,j) = tmp_image;
  end

  imdb = final_merged_imdb;


% -------------------------------------------------------------------------
function imdb = getDenselyDownProjectedImdb(imdb, number_of_projection_layers, projection_layer_type, number_of_non_linear_layers, non_linear_layer_type, projected_dim)
% -------------------------------------------------------------------------
  assert(number_of_non_linear_layers <= number_of_projection_layers);

  vectorized_imdb = getVectorizedImdb(imdb);
  original_dim = size(vectorized_imdb.images.data, 2);
  % projected_dim = original_dim;
  non_linear_layer_count = 0;
  for layer_count = 1 : number_of_projection_layers
    % SCHEME 1: 100 -> 25 -> 25 -> 25
    if layer_count == 1
      tmp_dim = original_dim; % this is original_dim if there's only 1 layer, or an evolving dim as we project further and further
    else
      tmp_dim = projected_dim; % this is original_dim if there's only 1 layer, or an evolving dim as we project further and further
      % TODO remove me:
      tmp_dim = 2048;
    end

    % % SCHEME 2: 100 -> 100 -> 100 -> 25
    % if layer_count ~= number_of_projection_layers
    %   tmp_dim = original_dim; % this is original_dim if there's only 1 layer, or an evolving dim as we project further and further
    % else
    %   tmp_dim = projected_dim; % this is original_dim if there's only 1 layer, or an evolving dim as we project further and further
    % end






    if strfind(projection_layer_type, 'dense-gaussian-indep-row')
      tmp_std = 1;
      random_projection_matrix = tmp_std .* randn(projected_dim, tmp_dim);
      % random_projection_matrix = tmp_std .* randn(projected_dim, tmp_dim) / sqrt(projected_dim);
      % random_projection_matrix = randn(projected_dim, tmp_dim) / 1000000 + 1;
      % random_projection_matrix = randn(projected_dim, tmp_dim) / 10000;
      % random_projection_matrix = randn(projected_dim, tmp_dim) / 100;
      % random_projection_matrix = ones(projected_dim, tmp_dim);

      % size(random_projection_matrix);


    elseif strfind(projection_layer_type, 'dense-gaussian-rotated-row')
      tmp_std = 1;
      random_projection_matrix = zeros(projected_dim, tmp_dim);
      random_projection_vector = tmp_std .* randn(projected_dim, 1);
      % random_projection_vector = tmp_std .* randn(projected_dim, 1) / sqrt(projected_dim);
      % random_projection_vector = randn(projected_dim, 1) / 1000000 + 1;
      % random_projection_vector = randn(projected_dim, 1) / 10000;
      % random_projection_vector = randn(projected_dim, 1) / 100;
      % random_projection_vector = ones(projected_dim, 1);
      for i = 1 : tmp_dim
        random_projection_matrix(:, i) = circshift(random_projection_vector, i - 1);
      end


    elseif strfind(projection_layer_type, 'sparse-gaussian')
      dim_kernel = str2num(getStringParameterStartingAtIndex(projection_layer_type, 17));
      [random_projection_mask, padded_imdb] = getRandomProjectionMaskForImdb(imdb, dim_kernel);

      assert(size(random_projection_mask, 3) == size(padded_imdb.images.data, 3));

      for channel_counter = 1:size(random_projection_mask, 3)
        if strfind(projection_layer_type, 'indep-row')
          for i = 1 : size(random_projection_mask, 1) % in each row, populate a new vector of random values in the mask
            tmp_coefficients = getCoefficientsForMask(dim_kernel);
            mask_on_indices_in_row = find(random_projection_mask(i,:,channel_counter) == 1);
            random_projection_mask(i,mask_on_indices_in_row,channel_counter) = tmp_coefficients;
          end
        elseif strfind(projection_layer_type, 'rotated-row')
          tmp_coefficients = getCoefficientsForMask(dim_kernel);
          for i = 1 : size(random_projection_mask, 1) % in each row, populate a new vector of random values in the mask
            mask_on_indices_in_row = find(random_projection_mask(i,:,channel_counter) == 1);
            random_projection_mask(i,mask_on_indices_in_row,channel_counter) = tmp_coefficients;
          end
        else
          throwException('[ERROR] sparse projection_layer_type unrecognized.');
        end
      end

      random_projection_matrix = random_projection_mask;
      imdb = padded_imdb;


    elseif strfind(projection_layer_type, 'dense-log-normal')
      lognormal_mean = -0.702;
      lognormal_var = 0.9355;
      mu = log( (lognormal_mean ^ 2) / sqrt(lognormal_var + lognormal_mean ^ 2));
      sigma = sqrt( log( lognormal_var / (lognormal_mean ^ 2) + 1));
      vectorized_generated_samples = lognrnd(mu, sigma, 1, (original_dim)^2);
      random_projection_matrix = reshape(generated_samples, projected_dim, tmp_dim);
    else
      throwException('[ERROR] projection_layer_type unrecognized.');
    end

    % result below is 4D imdb
    imdb = projectImdbUsingMatrix(imdb, random_projection_matrix);

    if non_linear_layer_count < number_of_non_linear_layers
      switch non_linear_layer_type
        case 'relu'
          imdb.images.data(imdb.images.data < 0) = 0;
        case 'pooling-max-vector-width-9-stride-1'
          imdb = getImdbAfterPerformingVectorizedPooling(imdb, projected_dim, 9, 1, @max);
        case 'pooling-max-vector-width-9-stride-4'
          imdb = getImdbAfterPerformingVectorizedPooling(imdb, projected_dim, 9, 4, @max);
        case 'pooling-avg-vector-width-4-stride-4'
          imdb = getImdbAfterPerformingVectorizedPooling(imdb, projected_dim, 4, 4, @mean);
        case 'pooling-min-vector-width-4-stride-4'
          imdb = getImdbAfterPerformingVectorizedPooling(imdb, projected_dim, 4, 4, @min);
        case 'pooling-max-vector-width-4-stride-4'
          imdb = getImdbAfterPerformingVectorizedPooling(imdb, projected_dim, 4, 4, @max);
        case 'pooling-min-vector-width-16-stride-16'
          imdb = getImdbAfterPerformingVectorizedPooling(imdb, projected_dim, 16, 16, @min);
        case 'pooling-max-vector-width-16-stride-16'
          imdb = getImdbAfterPerformingVectorizedPooling(imdb, projected_dim, 16, 16, @max);
        case 'pooling-max-vector-width-25-stride-25'
          imdb = getImdbAfterPerformingVectorizedPooling(imdb, projected_dim, 25, 25, @max);
        case 'pooling-max-2x2-stride-2-pad-0000'
          imdb = convert4DImdbOnVectorsInto4DImdbOnImages(imdb);
          imdb.images.data = vl_nnpool(single(imdb.images.data), [2, 2], 'Stride', 2, 'Pad', [0 0 0 0], 'method', 'max'); % great, we can apply vl_nnpool on top of a 4D data...
        case 'pooling-max-3x3-stride-2-pad-0101'
          imdb = convert4DImdbOnVectorsInto4DImdbOnImages(imdb);
          imdb.images.data = vl_nnpool(single(imdb.images.data), [3, 3], 'Stride', 2, 'Pad', [0 1 0 1], 'method', 'max');
        case 'relu_w_pooling-max-3x3-stride-2-pad-0101'
          imdb = convert4DImdbOnVectorsInto4DImdbOnImages(imdb);
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
function [random_projection_mask, padded_imdb] = getRandomProjectionMaskForImdb(imdb, dim_kernel)
% -------------------------------------------------------------------------
  assert(size(imdb.images.data, 1) == size(imdb.images.data, 2));
  dim_image = size(imdb.images.data, 1);
  padding = (dim_kernel - 1) / 2;
  random_projection_mask = createToeplitzMask(dim_image, dim_kernel, false);
  random_projection_mask = repmat(random_projection_mask, [1,1,size(imdb.images.data, 3)]);
  size(random_projection_mask)
  padded_imdb = padImdbUsingPadding(imdb, [padding, padding, padding, padding]);
  assert(size(padded_imdb.images.data, 1) == size(padded_imdb.images.data, 2));
  dim_padded_image = size(padded_imdb.images.data, 1);
  assert(isequal([size(random_projection_mask, 1), size(random_projection_mask, 2), size(random_projection_mask, 3)], [dim_image^2, dim_padded_image^2, size(padded_imdb.images.data, 3)]), 'something went wrong with the dimensions.');


% -------------------------------------------------------------------------
function tmp_coefficients = getCoefficientsForMask(dim_kernel)
% -------------------------------------------------------------------------
  tmp_coefficients = randn(dim_kernel^2, 1);
  % tmp_coefficients = tmp_coefficients / 1000000 + 1;
  % tmp_coefficients = tmp_coefficients / 10000;
  % tmp_coefficients = tmp_coefficients / 100;

  % tmp_coefficients = ones(dim_kernel^2, 1);
  % tmp_coefficients = zeros(dim_kernel^2, 1);


% -------------------------------------------------------------------------
function imdb = convert4DImdbOnVectorsInto4DImdbOnImages(imdb)
% -------------------------------------------------------------------------
  image_side_length = sqrt(size(imdb.images.data, 1));
  assert(floor(image_side_length) == image_side_length, 'expected image to be square for perfect image pooling.');
  assert(size(imdb.images.data, 3) == 1, 'no channels allowed in this code.');

  number_of_samples = size(imdb.images.data, 4);
  tmp_imdb = imdb;
  tmp_imdb.images.data = zeros(image_side_length, image_side_length, 1, number_of_samples);
  for i = 1 : number_of_samples
    tmp_imdb.images.data(:,:,1,i) = reshape(imdb.images.data(:,:,:,i), image_side_length, image_side_length, 1);
  end

  imdb = tmp_imdb;


% -------------------------------------------------------------------------
function imdb = getImdbAfterPerformingVectorizedPooling(imdb, projected_dim, pooling_width, pooling_stride, fh_operation)
% -------------------------------------------------------------------------
  % pooling_width = 9;
  % pooling_stride = 1;
  assert(pooling_width >= pooling_stride);

  % tmp_random_min_max_order = sign(randn(1, 100000));

  number_of_channels = size(imdb.images.data, 3);
  number_of_samples = size(imdb.images.data, 4);

  tmp_imdb = imdb;
  % tmp_imdb.images.data = zeros(projected_dim, 1, number_of_channels, number_of_samples);
  tmp_imdb.images.data = [];
  for j = 1 : number_of_samples
    for k = 1 : number_of_channels
      input_sample = imdb.images.data(:,:,k,j);
      pooled_sample = [];
      assert(size(input_sample, 1) == projected_dim); % TODO: you can't go from 784 -> 64, because the code I've written for sparse topelitz does convolutions that retain image size!
      assert(size(input_sample, 2) == 1);
      assert(size(input_sample, 3) == 1);
      assert(mod(projected_dim, pooling_stride) == 0);
      right_padded_input_sample = cat(2, reshape(input_sample, 1, []), zeros(1, pooling_width - pooling_stride));
      for ll = 1 : pooling_stride : projected_dim
        pooled_sample(end+1) = fh_operation(right_padded_input_sample(ll : ll + pooling_width - 1));
        % if tmp_random_min_max_order(k) == 1
        %   pooled_sample(end+1) = max(right_padded_input_sample(k : k + pooling_width - 1));
        % elseif tmp_random_min_max_order(k) == -1
        %   pooled_sample(end+1) = min(right_padded_input_sample(k : k + pooling_width - 1));
        % else
        %   throwException('[ERROR] values should only be +1 / -1.');
        % end
      end
      % keyboard
      % tmp_imdb.images.data(:,1,k,j) = pooled_sample;
      if k == 1 && j == 1 % starting off
        tmp_imdb.images.data(:,:,k,end) = pooled_sample;
      elseif k == 1 % when going onto a new sample
        tmp_imdb.images.data(:,:,k,end+1) = pooled_sample;
      else % when adding additional channels of a sample
        tmp_imdb.images.data(:,:,k,end) = pooled_sample;
      end
      % if k == 1 && j == 1
      %   tmp_imdb.images.data(:,:,k,end) = pooled_sample;
      % else
      %   tmp_imdb.images.data(:,:,:,end+1) = pooled_sample;
      % end
    end
  end
  size(imdb.images.data)
  size(tmp_imdb.images.data)
  assert(prod(size(imdb.images.data)) == pooling_stride * prod(size(tmp_imdb.images.data)));
  imdb = tmp_imdb;


% -------------------------------------------------------------------------
function imdb = projectImdbUsingMatrix(imdb, projection_matrix)
% -------------------------------------------------------------------------
  sample_image = imdb.images.data(:,:,:,1);
  number_of_projection_matrix_channels = size(projection_matrix, 3);
  assert(prod(size(sample_image)) == size(projection_matrix, 2) * number_of_projection_matrix_channels);

  % IMPORTANT: projection matrix is either
  %    single channel: each sample                 of the imdb should be vectorized, and projected using the single channel of projection_matrix
  %    multi- channel: each channel of each sample of the imdb should be vectorized, and projected using each channel of projection_matrix

  if number_of_projection_matrix_channels == 1
    vectorized_original_imdb = getVectorizedImdb(imdb);

    projected_dim = size(projection_matrix, 1);
    number_of_samples = size(vectorized_original_imdb.images.data, 1);

    all_data_original_vectorized = vectorized_original_imdb.images.data;
    all_data_projected_vectorized = (projection_matrix * all_data_original_vectorized')';
    vectorized_projected_imdb = vectorized_original_imdb;
    vectorized_projected_imdb.images.data = all_data_projected_vectorized;

    imdb = get4DImdb(vectorized_projected_imdb, projected_dim, 1, 1, number_of_samples);
    % imdb = get4DImdb(vectorized_projected_imdb, sqrt(projected_dim), sqrt(projected_dim), 1, number_of_samples);
  else
    for channel_counter = 1 : number_of_projection_matrix_channels
      per_channel_original_imdb = imdb;
      per_channel_original_imdb.images.data = per_channel_original_imdb.images.data(:,:,channel_counter,:);

      vectorized_per_channel_original_imdb = getVectorizedImdb(per_channel_original_imdb);

      per_channel_projected_dim = size(projection_matrix, 1);
      number_of_samples = size(vectorized_per_channel_original_imdb.images.data, 1);

      all_data_per_channel_original_vectorized = vectorized_per_channel_original_imdb.images.data;
      all_data_per_channel_projected_vectorized = (projection_matrix(:,:,channel_counter) * all_data_per_channel_original_vectorized')';
      vectorized_per_channel_projected_data = all_data_per_channel_projected_vectorized;

      if channel_counter == 1
        vectorized_per_channel_projected_imdb = vectorized_per_channel_original_imdb; % to copy over labels and set
        vectorized_all_channel_projected_imdb.images.data = zeros( ...
          size(vectorized_per_channel_projected_data, 1), ...
          size(vectorized_per_channel_projected_data, 2) * number_of_projection_matrix_channels);
      end
      vectorized_all_channel_projected_imdb.images.data(:,1 + (channel_counter - 1) * 1024 : channel_counter * 1024) = vectorized_per_channel_projected_data;
    end
    vectorized_projected_imdb = vectorized_all_channel_projected_imdb;

    imdb = get4DImdb(vectorized_projected_imdb, per_channel_projected_dim, 1, number_of_projection_matrix_channels, number_of_samples);
    % imdb = get4DImdb(vectorized_projected_imdb, sqrt(projected_dim), sqrt(projected_dim), 1, number_of_samples);
  end



































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
    angle_separation_matrix = V;
  else
    angle_separation_matrix = eye(size(M_D));
  end

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

  if ~isfield(imdb, 'name')
    % imdb.name = 'mnist-784-two-class-5-0';
    imdb.name = 'mnist-784-multi-class-subsampled-balanced-250';
  end
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
      if mod(k, 10000) == 0
        for j = 0:log10(k - 1) + (3 + numel(num2str(length(input_set))))
          fprintf('\b'); % delete previous counter display
        end
        fprintf('%d / %d', k, length(input_set));
      end
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



























