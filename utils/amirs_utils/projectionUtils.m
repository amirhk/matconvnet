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

  fh.projectAndSaveImdbThroughNetworkArch = @projectAndSaveImdbThroughNetworkArch;
  fh.getProjectedImdbSamplesOnNetworkArch = @getProjectedImdbSamplesOnNetworkArch;
  fh.getNetworkObjectFromNetworkArchWithoutLearningRate = @getNetworkObjectFromNetworkArchWithoutLearningRate;

% -------------------------------------------------------------------------
function projected_imdb = projectAndSaveImdbThroughNetworkArch(dataset, posneg_balance, network_arch, forward_pass_depth)
% -------------------------------------------------------------------------
  % get imdb
  tmp_opts.dataset = dataset;
  tmp_opts.posneg_balance = posneg_balance;
  input_imdb = loadSavedImdb(tmp_opts);
  train_imdb = filterImdbForSet(input_imdb, 1);
  test_imdb = filterImdbForSet(input_imdb, 3);




  % train_imdb.images.data = ones(5,5,1,2, 'single');
  % train_imdb.images.data(:,:,:,1) = zeros(5,5,1,1, 'single');
  % train_imdb.images.labels = [1,2];
  % train_imdb.images.set = [3,3];

  % test_imdb.images.data = 3 * ones(5,5,1,2, 'single');
  % test_imdb.images.data(:,:,:,1) = 3 * zeros(5,5,1,1, 'single');
  % test_imdb.images.labels = [1,2];
  % test_imdb.images.set = [3,3];


  % get net
  net = getNetworkObjectFromNetworkArchWithoutLearningRate(dataset, network_arch);

  % train_imdb.images.data
  % all_train_samples_forward_pass_results = getProjectedImdbSamplesOnNet(train_imdb, net, forward_pass_depth)
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

  imdb = projected_imdb;
  afprintf(sprintf('[INFO] Saving imdb...\n'));
  save_file_name = sprintf( ...
    'saved-projected-%s-%s-through-%s', ...
    dataset, ...
    posneg_balance, ...
    network_arch);
  % save(save_file_name, 'imdb');
  % save(save_file_name, 'imdb', '-v7.3');
  save(save_file_name, 'imdb', '-v7.3', '-nocompression');


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
function net = getNetworkObjectFromNetworkArchWithoutLearningRate(dataset, network_arch)
% -------------------------------------------------------------------------

  opts.general.dataset = dataset;
  opts.general.network_arch = network_arch;
  opts.net.weight_init_source = 'gen';
  opts.net.weight_init_sequence = {'compRand', 'compRand', 'compRand', 'compRand', 'compRand'};
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

