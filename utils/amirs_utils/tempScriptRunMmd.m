% -------------------------------------------------------------------------
function tempScriptRunMmd()
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
  %                                                                 Get IMDBs
  % -------------------------------------------------------------------------
  % dataset = 'cifar';
  % posneg_balance = 'whatever';
  % dataset = 'cifar-multi-class-subsampled';
  % posneg_balance = 'balanced-266';
  % dataset = 'cifar-two-class-deer-truck';

  % dataset = 'gaussian-5D-160-train-40-test';
  % dataset = 'gaussian-10D-160-train-40-test';
  % dataset = 'gaussian-25D-160-train-40-test';
  % dataset = 'gaussian-50D-160-train-40-test';

  % dataset = 'gaussian-5D-400-train-100-test';
  % dataset = 'gaussian-10D-400-train-100-test';
  % dataset = 'gaussian-25D-400-train-100-test';
  dataset = 'gaussian-50D-400-train-100-test';

  posneg_balance = 'balanced-38';

  [~, experiments] = setupExperimentsUsingProjectedImbds(dataset, posneg_balance, 1);

  for i = 1 : numel(experiments)
    unique_labels = unique(experiments{i}.imdb.images.labels);
    number_of_classes = numel(unique_labels);
    counter = 1;
    vals = [];
    bounds = [];
    for j = 1:number_of_classes
      for k = j+1:number_of_classes
        class_label_1 = unique_labels(j);
        class_label_2 = unique_labels(k);
        afprintf(sprintf( ...
          '[INFO] Running MMD on pair %d / %d (class 1: %d, class 2: %d) \t ', ...
          counter, ...
          number_of_classes * (number_of_classes - 1) / 2, ...
          class_label_1, ...
          class_label_2));
        % [experiments{i}.H, experiments{i}.info] = runKmdOnImdb(experiments{i}.imdb, class_label_1, class_label_2);
        [~, tmp] = runKmdOnImdb(experiments{i}.imdb, class_label_1, class_label_2);
        vals(end + 1) = tmp.mmd.val;
        bounds(end + 1) = tmp.mmd.bound;
        counter = counter + 1;
      end
    end
    experiments{i}.mmd_val_mean = mean(vals);
    experiments{i}.mmd_val_std = std(vals);
    experiments{i}.mmd_bound_mean = mean(bounds);
    experiments{i}.mmd_bound_std = std(bounds);
    printConsoleOutputSeparator();
  end

  for i = 1 : numel(experiments)
    afprintf(sprintf( ...
      '[INFO] MMD Results for `%s`: \t\t val = %.6f +/- %.6f, bound = %.6f +/- %.6f\n\n', ...
      experiments{i}.title, ...
      experiments{i}.mmd_val_mean, ...
      experiments{i}.mmd_val_std, ...
      experiments{i}.mmd_bound_mean, ...
      experiments{i}.mmd_bound_std));
  end

% -------------------------------------------------------------------------
function [H, info] = runKmdOnImdb(imdb, class_label_1, class_label_2)
% -------------------------------------------------------------------------
  % keyboard
  % data_train = imdb.images.data(:,:,:,imdb.images.set == 1);
  % labels_train = imdb.images.labels(imdb.images.set == 1);
  % sample_size = size(data_train, 1) * size(data_train, 2) * size(data_train, 3);
  % samples = reshape(data_train, sample_size, [])';

  indices_for_2_classes = bsxfun(@and, bsxfun(@or, imdb.images.labels == class_label_1, imdb.images.labels == class_label_2), imdb.images.set == 1);
  imdb.images.labels(indices_for_2_classes);

  data_train = imdb.images.data(:,:,:,indices_for_2_classes);
  labels_train = imdb.images.labels(indices_for_2_classes);

  sample_size = size(data_train, 1) * size(data_train, 2) * size(data_train, 3);
  samples = reshape(data_train, sample_size, [])';


  X = samples;
  % labels = (-1).^labels_train';
  labels_train(labels_train == class_label_1) = 1;
  labels_train(labels_train == class_label_2) = -1;
  labels = labels_train';
  % keyboard
  [H,info] = kmd(X,labels);















