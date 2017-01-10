% -------------------------------------------------------------------------
function fh = imdbMultiClassUtils()
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

  % assign function handles so we can call these local functions from elsewhere
  fh.getImdbInfo = @getImdbInfo;

% -------------------------------------------------------------------------
function [ ...
  data_train_per_class, ...
  data_train_count_per_class, ...
  data_train_indices_per_class, ...
  data_test_per_class, ...
  data_test_count_per_class, ...
  data_test_indices_per_class, ...
  labels_test] = getImdbInfo(imdb, debug_flag)
% -------------------------------------------------------------------------
  % enforce row vector before doing bsxfun
  imdb.images.labels = reshape(imdb.images.labels, 1, prod(size(imdb.images.labels)));
  imdb.images.set = reshape(imdb.images.set, 1, prod(size(imdb.images.set)));


  unique_classes = unique(imdb.images.labels);

  for class_number = unique_classes
    % train
    data_train_indices_per_class{class_number} = bsxfun(@and, imdb.images.labels == class_number, imdb.images.set == 1);
    data_train_count_per_class{class_number} = sum(data_train_indices_per_class{class_number});
    data_train_per_class{class_number} = imdb.images.data(:,:,:,data_train_indices_per_class{class_number});
    % test
    data_test_indices_per_class{class_number} = bsxfun(@and, imdb.images.labels == class_number, imdb.images.set == 3);
    data_test_count_per_class{class_number} = sum(data_test_indices_per_class{class_number});
    data_test_per_class{class_number} = imdb.images.data(:,:,:,data_test_indices_per_class{class_number});
  end

  if debug_flag
    afprintf(sprintf('[INFO] imdb info:\n'));
    afprintf(sprintf('[INFO] TRAINING SET:\n'));
    afprintf(sprintf('[INFO] total: %d\n', sum([data_train_count_per_class{:}], 2)), 1);
    for class_number = unique_classes
      afprintf(sprintf('[INFO] class #%d: %d\n', class_number, data_train_count_per_class{class_number}), 1);
    end
    afprintf(sprintf('[INFO] TESTING SET:\n'));
    afprintf(sprintf('[INFO] total: %d\n', sum([data_test_count_per_class{:}], 2)), 1);
    for class_number = unique_classes
      afprintf(sprintf('[INFO] class #%d: %d\n', class_number, data_test_count_per_class{class_number}), 1);
    end
  end
