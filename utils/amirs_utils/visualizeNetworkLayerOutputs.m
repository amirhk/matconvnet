% -------------------------------------------------------------------------
function visualizeNetworkLayerOutputs(input_opts)
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

  % -------------------------------------------------------------------------
  %                                                              opts.general
  % -------------------------------------------------------------------------
  opts.general.dataset = getValueFromFieldOrDefault(input_opts, 'dataset', 'cifar');
  opts.general.network_arch = getValueFromFieldOrDefault(input_opts, 'network_arch', 'lenet');

  % -------------------------------------------------------------------------
  %                                                                 opts.imdb
  % -------------------------------------------------------------------------
  opts.imdb.imdb = getValueFromFieldOrDefault(input_opts, 'imdb', struct()); % may optionally pass in the imdb

  % -------------------------------------------------------------------------
  %                                                                opts.train
  % -------------------------------------------------------------------------
  opts.train.gpus = ifNotMacSetGpu(getValueFromFieldOrDefault(input_opts, 'gpus', 1));

  % -------------------------------------------------------------------------
  %                                                                opts.paths
  % -------------------------------------------------------------------------
  opts.paths.time_string = sprintf('%s', datestr(datetime('now', 'Format', 'd-MMM-y-HH-mm-ss')));
  opts.paths.experiment_parent_dir = getValueFromFieldOrDefault( ...
    input_opts, ...
    'experiment_parent_dir', ...
    fullfile(vl_rootnn, 'experiment_visualizations'));
  opts.paths.experiment_dir = fullfile(opts.paths.experiment_parent_dir, sprintf( ...
    'cnn-%s-%s-%s-%s-%s', ...
    opts.paths.time_string, ...
    opts.general.dataset));
  if ~exist(opts.paths.experiment_dir)
    mkdir(opts.paths.experiment_dir);
  end
  opts.paths.options_file_path = fullfile(opts.paths.experiment_dir, '_options.txt');
  opts.paths.results_file_path = fullfile(opts.paths.experiment_dir, '_results.txt');

  % -------------------------------------------------------------------------
  %                                                    save experiment setup!
  % -------------------------------------------------------------------------
  opts_copy = opts;
  % opts_copy.net = rmfield(opts_copy.net, 'net');
  % opts_copy.imdb = rmfield(opts_copy.imdb, 'imdb');
  opts_copy.net.net = '< too large to print net >';
  opts_copy.imdb.imdb = '< too large to print imdb >';
  saveStruct2File(opts_copy, opts.paths.options_file_path, 0);

  % -------------------------------------------------------------------------
  %                                                                     beef!
  % -------------------------------------------------------------------------
  % get rid of extra crap
  fh_projection_utils = projectionUtils();
  net = fh_projection_utils.getProjectionNetworkObject( ...
    opts.general.dataset, ...
    opts.general.network_arch);
  imdb = opts.imdb.imdb;

  % save input sample
  sample_number = 1;
  h = figure;
  sample_image = imdb.images.data(:,:,:, sample_number);
  assert(numel(size(sample_image)) == 3); % assert sample only has 3 dimensions
  number_of_image_channels = size(sample_image, 3);
  file_name = 'Input Layer';
  for jj = 1:number_of_image_channels
    subplot(1, number_of_image_channels, jj), imshow(sample_image(:,:,jj),[]);
  end
  saveas(h, fullfile(opts.paths.experiment_dir, sprintf('%s.png%', file_name)));


  imdb = filterImdbForSet(imdb, 1, 3); % TODO: this is set to train.... should this change?????
  conv_count = 0;
  for ii = 1:numel(net.layers)
    if strcmp(net.layers{ii}.type, 'conv')
      conv_count = conv_count + 1;
      file_name = sprintf('Layer #%d - Conv #%d', ii, conv_count);

      % get forward pass matrices for all samples at certain depth
      all_samples_forward_pass_results = ...
        fh_projection_utils.getProjectedImdbSamplesOnNetworkArch(imdb, net, ii);

      % monkey work
      number_of_layer_outputs = size(net.layers{ii}.weights{1}, 4);
      subplot_width = 8; % TODO: this can be better... but whatever....
      assert(mod(number_of_layer_outputs, subplot_width) == 0); % so equal number of subplots in each row...
      subplot_height = number_of_layer_outputs / subplot_width;
      assert(subplot_height * subplot_width == size(all_samples_forward_pass_results, 3));

      % plot and save
      h = figure;
      % title(file_name);
      for jj = 1:number_of_layer_outputs
        subplot(subplot_height, subplot_width, jj), imshow(all_samples_forward_pass_results(:,:,jj,sample_number),[]);
      end
      saveas(h, fullfile(opts.paths.experiment_dir, sprintf('%s.png%', file_name)));
    end
  end
  close all


% % get imdb
% tmp_opts.dataset = 'mnist-two-class-9-4';
% tmp_opts.posneg_balance = 'balanced-38';
% tmp_opts.projection = 'no-projection';
% imdb = loadSavedImdb(tmp_opts, 0);

% % visualize
% input_opts.dataset = 'mnist-two-class-9-4';
% input_opts.imdb = imdb;
% input_opts.network_arch = 'TMP_NETWORK';
% visualizeNetworkLayerOutputs(input_opts)
