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

  fh.projectImdbThroughNetworkArch = @projectImdbThroughNetworkArch;
  fh.projectAndSaveImdbThroughNetworkArch = @projectAndSaveImdbThroughNetworkArch;
  fh.getProjectedImdbSamplesOnNetworkArch = @getProjectedImdbSamplesOnNetworkArch;
  fh.getNetworkObjectFromNetworkArchWithoutLearningRate = @getNetworkObjectFromNetworkArchWithoutLearningRate;

% % -------------------------------------------------------------------------
% function projected_imdb = projectAndSaveImdbThroughNetworkArch(dataset, posneg_balance, network_arch, larp_weight_init_type, forward_pass_depth)
% % -------------------------------------------------------------------------
%   projected_imdb = projectImdbThroughNetworkArch(dataset, posneg_balance, network_arch, larp_weight_init_type, forward_pass_depth)
%   imdb = projected_imdb;
%   afprintf(sprintf('[INFO] Saving imdb...\n'));
%   save_file_name = sprintf( ...
%     'saved-projected-%s-%s-through-%s-%s', ...
%     dataset, ...
%     posneg_balance, ...
%     network_arch, ...
%     larp_weight_init_type);
%   % save(save_file_name, 'imdb');
%   save(save_file_name, 'imdb', '-v7.3');
%   % save(save_file_name, 'imdb', '-v7.3', '-nocompression');

% -------------------------------------------------------------------------
function projected_imdb = projectImdbThroughNetworkArch(dataset, posneg_balance, larp_network_arch, larp_weight_init_sequence, forward_pass_depth)
% -------------------------------------------------------------------------
  % get imdb
  tmp_opts.dataset = dataset;
  tmp_opts.posneg_balance = posneg_balance;
  input_imdb = loadSavedImdb(tmp_opts);
  train_imdb = filterImdbForSet(input_imdb, 1);
  test_imdb = filterImdbForSet(input_imdb, 3);


  % train_imdb.images.data = zeros(5,5,3,10, 'single');
  % for i = 1:10
  %   train_imdb.images.data(:,:,:,i) = i * ones(5,5,3,1, 'single');
  % end
  % train_imdb.images.labels = 1:10;
  % train_imdb.images.set = 3 * ones(1,10);

  % train_imdb.images.data = 10 * ones(5,5,3,2, 'single');
  % train_imdb.images.data(:,:,:,1) = 1 * ones(5,5,3,1, 'single');
  % train_imdb.images.labels = [1,2];
  % train_imdb.images.set = [3,3];

  % test_imdb.images.data = 3 * ones(5,5,3,2, 'single');
  % test_imdb.images.data(:,:,:,1) = 3 * zeros(5,5,3,1, 'single');
  % test_imdb.images.labels = [1,2];
  % test_imdb.images.set = [3,3];


  % get net
  net = getNetworkObjectFromNetworkArchWithoutLearningRate(dataset, larp_network_arch, larp_weight_init_sequence);
  if forward_pass_depth == -1
    forward_pass_depth = numel(net.layers); % +1 necessary?.... guess not!
  end

  % train_imdb.images.data
  % all_train_samples_forward_pass_results = getProjectedImdbSamplesOnNet(train_imdb, net, 1)
  % % all_train_samples_forward_pass_results = getProjectedImdbSamplesOnNet(train_imdb, net, 2)
  % % all_train_samples_forward_pass_results = getProjectedImdbSamplesOnNet(train_imdb, net, 3)
  % % all_train_samples_forward_pass_results = getProjectedImdbSamplesOnNet(train_imdb, net, 4)
  % keyboard

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
function projected_samples = getProjectedImdbSamplesOnNet(imdb, net, depth)
% -------------------------------------------------------------------------
  assert(numel(find(imdb.images.set == 1)) == 0); % cnn_train only projects test data. sad, i know.
  [net, info] = cnnTrain(net, imdb, getBatch(), ...
    'forward_pass_only_mode', true, ...
    'forward_pass_only_depth', depth + 1, ... % +1 is critical because for a 3 layer network, cnn_train's res variable has 4 layers incl'd the input.
    'debug_flag', false, ...
    'continue', false, ...
    'num_epochs', 1, ...
    'train', [], ...
    'val', find(imdb.images.set == 3));
  projected_samples = info.all_samples_forward_pass_results;


% -------------------------------------------------------------------------
function net = getNetworkObjectFromNetworkArchWithoutLearningRate(dataset, larp_network_arch, larp_weight_init_sequence)
% -------------------------------------------------------------------------
  opts.general.dataset = dataset;
  opts.general.network_arch = larp_network_arch;
  opts.net.weight_init_source = 'gen';
  opts.net.weight_init_sequence = larp_weight_init_sequence;
  opts.train.learning_rate = [999*ones(1,1)]; % doesn't matter, as we're not training....
  network_opts = cnnInit(opts);
  net = network_opts.net;

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

