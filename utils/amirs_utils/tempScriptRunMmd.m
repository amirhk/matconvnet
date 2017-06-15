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
  % posneg_balance = 'balanced-707';
  dataset = 'cifar-two-class-deer-truck';
  posneg_balance = 'balanced-38';

  [~, experiments] = setupExperimentsUsingProjectedImbds(dataset, posneg_balance, 1);

  for i = 1 : numel(experiments)
    unique_labels = unique(experiments{i}.imdb.images.labels);
    number_of_unique_labels = numel(unique_labels);
    for j = 1:number_of_unique_labels
      for k = j:number_of_unique_labels
        class_label_1 = unique_labels(j);
        class_label_2 = unique_labels(k);
        disp(class_label_1);
        disp(class_label_2);
        % tic
        % [experiments{i}.H, experiments{i}.info] = runKmdOnImdb(experiments{i}.imdb, class_label_1, class_label_2);
        % toc
      end
    end
  end

  for i = 1 : numel(experiments)
    afprintf(sprintf( ...
      '[INFO] MMD Results for `%s`: \t\t val = %.6f, bound = %.6f\n\n', ...
      experiments{i}.title, ...
      experiments{i}.info.mmd.val, ...
      experiments{i}.info.mmd.bound));
  end

% -------------------------------------------------------------------------
function [H, info] = runKmdOnImdb(imdb, class_label_1, class_label_2)
% -------------------------------------------------------------------------
  data_train = imdb.images.data(:,:,:,imdb.images.set == 1);
  labels_train = imdb.images.labels(imdb.images.set == 1);
  sample_size = size(data_train, 1) * size(data_train, 2) * size(data_train, 3);
  samples = reshape(data_train, sample_size, [])';

  X = samples;
  labels = (-1).^labels_train';
  [H,info] = kmd(X,labels);
