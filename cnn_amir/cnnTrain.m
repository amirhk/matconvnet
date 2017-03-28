% -------------------------------------------------------------------------
function [net, info] = cnn_train(net, imdb, getBatch, varargin)
% -------------------------------------------------------------------------
  % CNN_TRAIN   Demonstrates training a CNN
  %    CNN_TRAIN() is an example learner implementing stochastic
  %    gradient descent with momentum to train a CNN. It can be used
  %    with different datasets and tasks by providing a suitable
  %    getBatch function.
  %
  %    The function automatically restarts after each training epoch by
  %    checkpointing.
  %
  %    The function supports training on CPU or on one or more GPUs
  %    (specify the list of GPU IDs in the `gpus` option). Multi-GPU
  %    support is relatively primitive but sufficient to obtain a
  %    noticable speedup.

  % Copyright (C) 2014-15 Andrea Vedaldi.
  % All rights reserved.
  %
  % This file is part of the VLFeat library and is made available under
  % the terms of the BSD license (see the COPYING file).

  opts.forward_pass_only_mode = false;
  opts.forward_pass_only_depth = -1;
  opts.debug_flag = true;
  opts.weight_init_source = 'gen'; % {'load' | 'gen'}
  opts.weight_init_sequence = {'compRand', 'compRand', 'compRand', 'compRand', 'compRand'};
  opts.backprop_depth = +inf;
  opts.batch_size = 256;
  opts.numSubBatches = 1;
  opts.train = [];
  opts.val = [];
  opts.num_epochs = 50;
  opts.gpus = []; % which GPU devices to use (none, one, or more)
  opts.learning_rate = 0.001;
  opts.continue = true;
  opts.experiment_dir = fullfile('data','exp');
  opts.conserveMemory = false;
  opts.sync = false;
  opts.prefetch = false;
  opts.cudnn = true;
  opts.weight_decay = 0.0005;
  opts.momentum = 0.9;
  opts.error_function = 'multiclass';
  opts.errorLabels = {};
  opts.plotDiagnostics = false;
  opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin');
  opts = vl_argparse(opts, varargin);

   if ~exist(opts.experiment_dir, 'dir'), mkdir(opts.experiment_dir); end
  if isempty(opts.train), opts.train = find(imdb.images.set==1); end
  if isempty(opts.val), opts.val = find(imdb.images.set==2); end
  if isnan(opts.train), opts.train = []; end

  % -------------------------------------------------------------------------
  %                                                    Network initialization
  % -------------------------------------------------------------------------

  evaluateMode = isempty(opts.train);
  trainingMode = isempty(opts.val);

  if ~evaluateMode
    for i=1:numel(net.layers)
      if isfield(net.layers{i}, 'weights')
        J = numel(net.layers{i}.weights);
        for j=1:J
          net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single');
        end
        if ~isfield(net.layers{i}, 'learning_rate')
          net.layers{i}.learning_rate = ones(1, J, 'single');
        end
        if ~isfield(net.layers{i}, 'weight_decay')
          net.layers{i}.weight_decay = ones(1, J, 'single');
        end
      end
      % Legacy code: will be removed
      if isfield(net.layers{i}, 'filters')
        net.layers{i}.momentum{1} = zeros(size(net.layers{i}.filters), 'single');
        net.layers{i}.momentum{2} = zeros(size(net.layers{i}.biases), 'single');
        if ~isfield(net.layers{i}, 'learning_rate')
          net.layers{i}.learning_rate = ones(1, 2, 'single');
        end
        if ~isfield(net.layers{i}, 'weight_decay')
          net.layers{i}.weight_decay = single([1 0]);
        end
      end
    end
  end

  % setup GPUs
  numGpus = numel(opts.gpus);
  if numGpus > 1
    if isempty(gcp('nocreate')),
      parpool('local',numGpus);
      spmd, gpuDevice(opts.gpus(labindex)), end
    end
  elseif numGpus == 1
    gpuDevice(opts.gpus);
  end
  if exist(opts.memoryMapFile), delete(opts.memoryMapFile); end

  % setup error calculation function
  if isstr(opts.error_function)
    switch opts.error_function
      case 'none'
        opts.error_function = @error_none;
      case 'multiclass'
        opts.error_function = @error_multiclass;
        if isempty(opts.errorLabels), opts.errorLabels = {'top1e', 'top5e'}; end
      case 'two-class'
        opts.error_function = @error_two_class;
        if isempty(opts.errorLabels), opts.errorLabels = {'top1e'}; end
      case 'binary'
        opts.error_function = @error_binary;
        if isempty(opts.errorLabels), opts.errorLabels = {'bine'}; end
      otherwise
        error('Uknown error function ''%s''', opts.error_function);
    end
  end

  % -------------------------------------------------------------------------
  %                                                        Train and validate
  % -------------------------------------------------------------------------

  if ~evaluateMode
    if ~opts.debug_flag
      afprintf(sprintf('[INFO] processing epoch #'));
    end
    for epoch=1:opts.num_epochs
      if ~opts.debug_flag
        for j = 0:log10(epoch - 1) + (3 + numel(num2str(opts.num_epochs)))
          fprintf('\b'); % delete previous counter display
        end
        fprintf('%d', epoch);
      end
      fprintf(' / %d', opts.num_epochs);

      learning_rate = opts.learning_rate(min(epoch, numel(opts.learning_rate)));

      % fast-forward to last checkpoint
      modelPath = @(ep) fullfile(opts.experiment_dir, sprintf('net-epoch-%d.mat', ep));
      modelFigPath = fullfile(opts.experiment_dir, 'net-train.pdf');
      if opts.continue
        if exist(modelPath(epoch),'file')
          if epoch == opts.num_epochs
            load(modelPath(epoch), 'net', 'info');
          end
          continue;
        end
        if epoch > 1
          if opts.debug_flag
            fprintf('resuming by loading epoch %d\n', epoch-1);
          end
          load(modelPath(epoch-1), 'net', 'info');
        end
      end

      % train one epoch and validate
      train = opts.train(randperm(numel(opts.train))); % shuffle
      val = opts.val;
      if numGpus <= 1
        [net,stats.train] = process_epoch(opts, getBatch, epoch, train, learning_rate, imdb, net);
        [~,stats.val] = process_epoch(opts, getBatch, epoch, val, 0, imdb, net);
      else
        spmd(numGpus)
          [net_, stats_train_] = process_epoch(opts, getBatch, epoch, train, learning_rate, imdb, net);
          [~, stats_val_] = process_epoch(opts, getBatch, epoch, val, 0, imdb, net_);
        end
        net = net_{1};
        stats.train = sum([stats_train_{:}],2);
        stats.val = sum([stats_val_{:}],2);
      end

      % save
      if evaluateMode
        sets = {'val'};
      elseif trainingMode
        sets = {'train'};
      else
        sets = {'train', 'val'};
      end
      for f = sets
        f = char(f);
        n = numel(eval(f));
        info.(f).speed(epoch) = n / stats.(f)(1) * max(1, numGpus);
        info.(f).objective(epoch) = stats.(f)(2) / n;
        info.(f).error(:,epoch) = stats.(f)(3 : 3 + numel(opts.errorLabels) - 1) / n;
      end
      if ~evaluateMode, save(modelPath(epoch), 'net', 'info'); end

      figure(1); clf;
      hasError = isa(opts.error_function, 'function_handle');
      subplot(1,1+hasError,1);
      if ~evaluateMode
        semilogy(1:epoch, info.train.objective, '.-', 'linewidth', 2);
        hold on;
      end
      if ~trainingMode
        semilogy(1:epoch, info.val.objective, '.--');
      end
      xlabel('training epoch'); ylabel('energy');
      grid on;
      h=legend(sets);
      set(h,'color','none');
      title('objective');
      if hasError
        subplot(1,2,2); leg = {};
        if ~evaluateMode
          plot(1:epoch, info.train.error', '.-', 'linewidth', 2);
          hold on;
          leg = horzcat(leg, strcat('train ', opts.errorLabels));
        end
        if ~trainingMode
          plot(1:epoch, info.val.error', '.--');
          leg = horzcat(leg, strcat('val ', opts.errorLabels));
        end
        set(legend(leg{:}),'color','none');
        grid on;
        xlabel('training epoch'); ylabel('error');
        title('error');
      end
      drawnow;
      print(1, modelFigPath, '-dpdf');
    end
    if ~opts.debug_flag
      fprintf('\n');
    end
  elseif ~opts.forward_pass_only_mode
    % only to be used for validation
    epoch = 1;
    val = opts.val;
    if numGpus <= 1
      [top_predictions, all_predictions, labels] = get_all_samples_predictions_from_network(opts, getBatch, epoch, val, 0, imdb, net);
    else
      spmd(numGpus)
        [top_predictions_, all_predictions_, labels_] = get_all_samples_predictions_from_network(opts, getBatch, epoch, val, 0, imdb, net);
      end
      % TODO: WARNING: because the returned predictions could be coming from
      % multiple GPUs, the ordering of the predicited class may be fucked!
      top_predictions = cat(2, top_predictions_{:});
      all_predictions = cat(2, all_predictions_{:});
      labels = cat(2, labels_{:});
    end
    info.top_predictions = top_predictions;
    info.all_predictions = all_predictions;
    info.labels = labels;
  else
    assert(opts.forward_pass_only_mode);
    assert(opts.forward_pass_only_depth > 0);
    epoch = 1;
    val = opts.val;
    if numGpus <= 1
      all_samples_forward_pass_results = get_resulting_forward_pass_matrix_from_network_for_all_samples(opts, getBatch, epoch, val, 0, imdb, net, opts.forward_pass_only_depth);
    else
      spmd(numGpus)
        all_samples_forward_pass_results_ = get_resulting_forward_pass_matrix_from_network_for_all_samples(opts, getBatch, epoch, val, 0, imdb, net, opts.forward_pass_only_depth);
      end
      % TODO: WARNING: because the returned resulting matrices could be coming
      % from multiple GPUs, the ordering of the predicited class may be fucked!
      all_samples_forward_pass_results = cat(4, all_samples_forward_pass_results_{:});
    end
    info.all_samples_forward_pass_results = all_samples_forward_pass_results;
  end

% -------------------------------------------------------------------------
function err = error_multiclass(opts, labels, res)
% -------------------------------------------------------------------------
  predictions = gather(res(end-1).x);
  [~,predictions] = sort(predictions, 3, 'descend');

  % be resilient to badly formatted labels
  if numel(labels) == size(predictions, 4)
    labels = reshape(labels,1,1,1,[]);
  end

  % skip null labels
  mass = single(labels(:,:,1,:) > 0);
  if size(labels,3) == 2
    % if there is a second channel in labels, used it as weights
    mass = mass .* labels(:,:,2,:);
    labels(:,:,2,:) = [];
  end

  error = ~bsxfun(@eq, predictions, labels);
  err(1,1) = sum(sum(sum(mass .* error(:,:,1,:))));
  err(2,1) = sum(sum(sum(mass .* min(error(:,:,1:5,:),[],3))));

% -------------------------------------------------------------------------
function err = error_two_class(opts, labels, res)
% -------------------------------------------------------------------------
  predictions = gather(res(end-1).x);
  [~,predictions] = sort(predictions, 3, 'descend');

  % be resilient to badly formatted labels
  if numel(labels) == size(predictions, 4)
    labels = reshape(labels,1,1,1,[]);
  end

  % skip null labels
  mass = single(labels(:,:,1,:) > 0);
  if size(labels,3) == 2
    % if there is a second channel in labels, used it as weights
    mass = mass .* labels(:,:,2,:);
    labels(:,:,2,:) = [];
  end

  error = ~bsxfun(@eq, predictions, labels);
  err(1,1) = sum(sum(sum(mass .* error(:,:,1,:)))); % top1

% -------------------------------------------------------------------------
function err = error_binaryclass(opts, labels, res)
% -------------------------------------------------------------------------
  predictions = gather(res(end-1).x);
  error = bsxfun(@times, predictions, labels) < 0;
  err = sum(error(:));

% -------------------------------------------------------------------------
function err = error_none(opts, labels, res)
% -------------------------------------------------------------------------
  err = zeros(0,1);

% -------------------------------------------------------------------------
function  [net_cpu,stats,prof] = process_epoch(opts, getBatch, epoch, subset, learning_rate, imdb, net_cpu)
% -------------------------------------------------------------------------
  % move CNN to GPU as needed
  numGpus = numel(opts.gpus);
  if numGpus >= 1
    net = vl_simplenn_move(net_cpu, 'gpu');
  else
    net = net_cpu;
    net_cpu = [];
  end

  % validation mode if learning rate is zero
  training = learning_rate > 0;
  if training, mode = 'training'; else, mode = 'validation'; end
  if nargout > 2, mpiprofile on; end

  numGpus = numel(opts.gpus);
  if numGpus >= 1
    one = gpuArray(single(1));
  else
    one = single(1);
  end
  res = [];
  mmap = [];
  stats = [];

  for t=1:opts.batch_size:numel(subset)
    if opts.debug_flag
      fprintf('%s: epoch %02d: batch %3d/%3d: ', mode, epoch, ...
              fix(t/opts.batch_size)+1, ceil(numel(subset)/opts.batch_size));
    end
    batch_size = min(opts.batch_size, numel(subset) - t + 1);
    batchTime = tic;
    num_done = 0;
    error = [];
    for s=1:opts.numSubBatches
      % get this image batch and prefetch the next
      batchStart = t + (labindex-1) + (s-1) * numlabs;
      batchEnd = min(t+opts.batch_size-1, numel(subset));
      batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd);
      [im, labels] = getBatch(imdb, batch);

      if opts.prefetch
        if s==opts.numSubBatches
          batchStart = t + (labindex-1) + opts.batch_size;
          batchEnd = min(t+2*opts.batch_size-1, numel(subset));
        else
          batchStart = batchStart + numlabs;
        end
        nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd);
        getBatch(imdb, nextBatch);
      end

      if numGpus >= 1
        im = gpuArray(im);
      end

      % evaluate CNN
      net.layers{end}.class = labels;
      if training, dzdy = one; else, dzdy = []; end
      res = vl_simplenn(net, im, dzdy, res, ...
                        'accumulate', s ~= 1, ...
                        'disableDropout', ~training, ...
                        'conserveMemory', opts.conserveMemory, ...
                        'backprop_depth', opts.backprop_depth, ...
                        'sync', opts.sync, ...
                        'cudnn', opts.cudnn);

      % accumulate training errors
      error = sum([ ...
        error, ...
        [ ...
          sum(double(gather(res(end).x))); ...
          reshape(opts.error_function(opts, labels, res),[],1); ...
        ] ...
      ], 2);
      num_done = num_done + numel(batch);
    end

    % gather and accumulate gradients across labs
    if training
      if numGpus <= 1
        [net,res] = accumulate_gradients(opts, learning_rate, batch_size, net, res);
      else
        if isempty(mmap)
          mmap = map_gradients(opts.memoryMapFile, net, res, numGpus);
        end
        write_gradients(mmap, net, res);
        labBarrier();
        [net,res] = accumulate_gradients(opts, learning_rate, batch_size, net, res, mmap);
      end
    end

    % print learning statistics
    batchTime = toc(batchTime);
    stats = sum([stats,[batchTime; error]],2); % works even when stats=[]
    speed = batch_size/batchTime;

    if opts.debug_flag
      fprintf(' %.2f s (%.1f data/s)', batchTime, speed);
      n = (t + batch_size - 1) / max(1,numlabs);
      fprintf(' obj:%.3g', stats(2)/n);
      for i=1:numel(opts.errorLabels)
        fprintf(' %s:%.3g', opts.errorLabels{i}, stats(i+2)/n);
      end
      fprintf(' [%d/%d]', num_done, batch_size);
      fprintf('\n');
    end

    % debug info
    if opts.plotDiagnostics && numGpus <= 1
      figure(2); vl_simplenn_diagnose(net,res); drawnow;
    end
  end

  if nargout > 2
    prof = mpiprofile('info');
    mpiprofile off;
  end

  if numGpus >= 1
    net_cpu = vl_simplenn_move(net, 'cpu');
  else
    net_cpu = net;
  end

% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients(opts, lr, batch_size, net, res, mmap)
% -------------------------------------------------------------------------
  for l=numel(net.layers):-1:1
    for j=1:min(numel(res(l).dzdw),1)
      thisDecay = opts.weight_decay * net.layers{l}.weight_decay(j);
      thisLR = lr * net.layers{l}.learning_rate(j);

      % accumualte from multiple labs (GPUs) if needed
      if nargin >= 6
        tag = sprintf('l%d_%d',l,j);
        tmp = zeros(size(mmap.Data(labindex).(tag)), 'single');
        for g = setdiff(1:numel(mmap.Data), labindex)
          tmp = tmp + mmap.Data(g).(tag);
        end
        res(l).dzdw{j} = res(l).dzdw{j} + tmp;
      end

      if isfield(net.layers{l}, 'weights')
        net.layers{l}.momentum{j} = ...
          opts.momentum * net.layers{l}.momentum{j} ...
          - thisDecay * net.layers{l}.weights{j} ...
          - (1 / batch_size) * res(l).dzdw{j};
        % net.layers{l}.momentum{j} = - (1 / batch_size) * res(l).dzdw{j};


        % net.layers{l}.weights{j} = (net.layers{l}.weights{j} + thisLR * net.layers{l}.momentum{j}); % I changed this line
        net.layers{l}.weights{j} = (net.layers{l}.weights{j} .* exp(thisLR * net.layers{l}.momentum{j})); % I changed this line

        % net.layers{l}.weights{j} = (net.layers{l}.weights{j} + thisLR * net.layers{l}.momentum{j}).*net.layers{l}.sparseMaps; % I changed this line (from Javad)
      else
        % Legacy code: to be removed
        if j == 1
          net.layers{l}.momentum{j} = ...
            opts.momentum * net.layers{l}.momentum{j} ...
            - thisDecay * net.layers{l}.filters ...
            - (1 / batch_size) * res(l).dzdw{j};
          net.layers{l}.filters = net.layers{l}.filters + thisLR * net.layers{l}.momentum{j};
        else
          net.layers{l}.momentum{j} = ...
            opts.momentum * net.layers{l}.momentum{j} ...
            - thisDecay * net.layers{l}.biases ...
            - (1 / batch_size) * res(l).dzdw{j};
          net.layers{l}.biases = net.layers{l}.biases + thisLR * net.layers{l}.momentum{j};
        end
      end
    end
  end

% -------------------------------------------------------------------------
function mmap = map_gradients(fname, net, res, numGpus)
% -------------------------------------------------------------------------
  format = {};
  for i=1:numel(net.layers)
    for j=1:numel(res(i).dzdw)
      format(end+1,1:3) = {'single', size(res(i).dzdw{j}), sprintf('l%d_%d',i,j)};
    end
  end
  format(end+1,1:3) = {'double', [3 1], 'errors'};
  if ~exist(fname) && (labindex == 1)
    f = fopen(fname,'wb');
    for g=1:numGpus
      for i=1:size(format,1)
        fwrite(f,zeros(format{i,2},format{i,1}),format{i,1});
      end
    end
    fclose(f);
  end
  labBarrier();
  mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true);

% -------------------------------------------------------------------------
function write_gradients(mmap, net, res)
% -------------------------------------------------------------------------
  for i=1:numel(net.layers)
    for j=1:numel(res(i).dzdw)
      mmap.Data(labindex).(sprintf('l%d_%d',i,j)) = gather(res(i).dzdw{j});
    end
  end

% -------------------------------------------------------------------------
function [all_samples_top_class_predictions, all_samples_all_class_predictions, all_labels] = get_all_samples_predictions_from_network(opts, getBatch, epoch, subset, learning_rate, imdb, net_cpu)
% -------------------------------------------------------------------------
  % validation mode if learning rate is zero
  % if nargout > 2, mpiprofile on; end

  % numGpus = numel(opts.gpus);
  % if numGpus >= 1
  %   one = gpuArray(single(1));
  % else
  %   one = single(1);
  % end

  % softmaxloss
  net_1 = net_cpu;
  % softmax
  net_2.layers = net_1.layers;
  net_2.layers{end}.type = 'softmax';
  afprintf(sprintf('Extracting `top`-class predictions based on `softmaxloss`\n'));
  [all_samples_top_class_predictions, ~, all_labels_1, ~] = tmpBeef(opts, getBatch, epoch, subset, learning_rate, imdb, net_1, 'softmaxloss', -1);
  all_samples_all_class_predictions = all_samples_top_class_predictions;

  % afprintf(sprintf('Extracting `all`-class predictions based on `softmax`\n'));
  % [~, all_samples_all_class_predictions, all_labels_2, ~] = tmpBeef(opts, getBatch, epoch, subset, learning_rate, imdb, net_2, 'softmax', -1);
  % assert(isequal(all_labels_1, all_labels_2));
  all_labels = all_labels_1;


% -------------------------------------------------------------------------
function [all_samples_forward_pass_results] = get_resulting_forward_pass_matrix_from_network_for_all_samples(opts, getBatch, epoch, subset, learning_rate, imdb, net_cpu, forward_pass_only_depth)
% -------------------------------------------------------------------------
  afprintf(sprintf('Extracting result of forward pass through network...\n'));
  [~, ~, ~, all_samples_forward_pass_results] = ...
    tmpBeef(opts, getBatch, epoch, subset, learning_rate, imdb, net_cpu, 'none', forward_pass_only_depth);


% -------------------------------------------------------------------------
function [all_samples_top_class_predictions, all_samples_all_class_predictions, all_labels, all_samples_forward_pass_results] = tmpBeef(opts, getBatch, epoch, subset, learning_rate, imdb, net_cpu, loss_layer_type, forward_pass_only_depth)
% -------------------------------------------------------------------------
  % move CNN to GPU as needed
  numGpus = numel(opts.gpus);
  if numGpus >= 1
    net = vl_simplenn_move(net_cpu, 'gpu');
  else
    net = net_cpu;
    net_cpu = [];
  end

  training = false;
  mode = 'validation';

  res = [];
  mmap = [];
  all_samples_top_class_predictions = [];
  all_samples_all_class_predictions = [];
  all_samples_forward_pass_results  = [];
  all_labels = [];
  if ~opts.debug_flag
    afprintf(sprintf('[INFO] processed     %d samples', 0), 1);
  end

  for t=1:opts.batch_size:numel(subset)
    if opts.debug_flag
      fprintf('%s: epoch %02d: batch %3d/%3d: ', mode, epoch, ...
              fix(t/opts.batch_size)+1, ceil(numel(subset)/opts.batch_size));
    end
    batch_size = min(opts.batch_size, numel(subset) - t + 1);
    num_done = 0;
    error = [];

    for s=1:opts.numSubBatches
      % get this image batch and prefetch the next
      batchStart = t + (labindex-1) + (s-1) * numlabs;
      batchEnd = min(t+opts.batch_size-1, numel(subset));
      batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd);
      [im, labels] = getBatch(imdb, batch);

      if opts.prefetch
        if s==opts.numSubBatches
          batchStart = t + (labindex-1) + opts.batch_size;
          batchEnd = min(t+2*opts.batch_size-1, numel(subset));
        else
          batchStart = batchStart + numlabs;
        end
        nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd);
        getBatch(imdb, nextBatch);
      end

      if numGpus >= 1
        im = gpuArray(im);
      end

      % evaluate CNN
      net.layers{end}.class = labels;
      % if training, dzdy = one; else, dzdy = []; end
      dzdy = [];
      res = vl_simplenn(net, im, dzdy, res, ...
                        'accumulate', s ~= 1, ...
                        'disableDropout', ~training, ...
                        'conserveMemory', opts.conserveMemory, ...
                        'backprop_depth', opts.backprop_depth, ...
                        'sync', opts.sync, ...
                        'cudnn', opts.cudnn);
      switch loss_layer_type
        case 'softmaxloss'
          batch_samples_all_class_predictions = gather(res(end-1).x);
          [~,batch_samples_top_class_predictions] = sort(batch_samples_all_class_predictions, 3, 'descend');
          all_samples_top_class_predictions = cat( ...
            2, ...
            all_samples_top_class_predictions, ...
            reshape(batch_samples_top_class_predictions(:,:,1,:), 1, []));
        case 'softmax'
          batch_samples_all_class_predictions = gather(res(end).x);
          all_samples_all_class_predictions = cat( ...
            2, ...
            all_samples_all_class_predictions, ...
            reshape(batch_samples_all_class_predictions(:,:,:,:), 2, []));
        case 'none'
          assert(numel(res) >= forward_pass_only_depth)
          batch_samples_all_class_predictions = gather(res(forward_pass_only_depth).x);
          all_samples_forward_pass_results = cat( ...
            4, ... % NOTE THE MERGE IN 4D... others reshape, we don't
            all_samples_forward_pass_results, ...
            batch_samples_all_class_predictions);
      end
      all_labels  = cat( ...
        2, ...
        all_labels, ...
        reshape(labels, 1, []));

      num_done = num_done + numel(batch);
    end

    if ~opts.debug_flag
      for j = 0:log10(batchEnd - 1) + 8 % + 8 because of ' samples'
        fprintf('\b'); % delete previous counter display
      end
      fprintf('%d samples', batchEnd);
    end
  end
  fprintf('\n');

  if numGpus >= 1
    net_cpu = vl_simplenn_move(net, 'cpu');
  else
    net_cpu = net;
  end











