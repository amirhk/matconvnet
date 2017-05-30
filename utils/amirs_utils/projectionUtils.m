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

  fh.getDenslyProjectedImdb = @getDenslyProjectedImdb;
  fh.projectImdbThroughNetwork = @projectImdbThroughNetwork;
  fh.getProjectionNetworkObject = @getProjectionNetworkObject;


% -------------------------------------------------------------------------
function projected_imdb = projectImdbThroughNetwork(imdb, net, forward_pass_depth)
% -------------------------------------------------------------------------
  % % get imdb
  % tmp_opts.dataset = dataset;
  % tmp_opts.posneg_balance = posneg_balance;
  input_imdb = imdb;
  train_imdb = filterImdbForSet(input_imdb, 1);
  test_imdb = filterImdbForSet(input_imdb, 3);

  % get net
  % net = getProjectionNetworkObject(dataset, larp_network_arch, larp_weight_init_sequence);
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
  assert(numel(find(imdb.images.set == 1)) == 0); % cnn_train only projects test data. sad, i know.
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
function imdb = getDenslyProjectedImdb(imdb)
% -------------------------------------------------------------------------
  projected_data = zeros(size(imdb.images.data));
  s1 = size(imdb.images.data, 1);
  s2 = size(imdb.images.data, 2);
  s3 = size(imdb.images.data, 3);
  s4 = size(imdb.images.data, 4);
  % use a consistent random projection matrix; output of random projection
  % should be same size as input(so we need N random projections, where N
  % is dimension of vectorized image)
  random_projection_matrix = randn(s1 * s2 * s3, s1 * s2 * s3) * 1/100;
  all_data_original_vectorized = reshape(imdb.images.data, s1 * s2 * s3, []); % [] = s4
  all_data_projected_vectorized = random_projection_matrix * all_data_original_vectorized; % 3072x3072 * 3072xnumber_of_images
  all_data_projected_matricized = reshape(all_data_projected_vectorized, s1, s2, s3, s4);
  imdb.images.data = all_data_projected_matricized;
  % for j = 1 : s4
  %   tmp = imdb.images.data(:,:,:,j);
  %   vectorized = reshape(tmp, s1 * s2 * s3, 1);
  %   projected = random_projection_matrix * vectorized;
  %   matricized = reshape(projected, s1, s2, s3);
  %   projected_data(:,:,:,j) = matricized;
  %   keyboard
  % end
  % imdb.images.data = projected_data;


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



























