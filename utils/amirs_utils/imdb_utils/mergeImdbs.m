% -------------------------------------------------------------------------
function imdb = mergeImdbs(train_imdb, test_imdb)
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
  assert( ...
    size(train_imdb.images.data, 1) * size(train_imdb.images.data, 2) * size(train_imdb.images.data, 3) == ...
    size(test_imdb.images.data, 1) * size(test_imdb.images.data, 2) * size(test_imdb.images.data, 3) );

  data_train = train_imdb.images.data;
  data_test = test_imdb.images.data;
  labels_train = reshape(train_imdb.images.labels, [], 1);
  labels_test = reshape(test_imdb.images.labels, [], 1);
  set_train = reshape(train_imdb.images.set, [], 1);
  set_test = reshape(test_imdb.images.set, [], 1);

  afprintf(sprintf('[INFO] Concatinating training data and testing data...\n'));
  data = single(cat(4, data_train, data_test));
  labels = single(cat(1, labels_train, labels_test));
  set = single(cat(1, set_train, set_test));
  afprintf(sprintf('[INFO] done!\n'));

  assert(length(labels) == length(set));
  total_number_of_samples = length(labels);

  % shuffle
  afprintf(sprintf('[INFO] Shuffling samples...\n'));
  ix = randperm(total_number_of_samples);
  imdb.images.data = data(:,:,:,ix);
  imdb.images.labels = labels(ix);
  imdb.images.set = set(ix);
  if isfield(train_imdb, 'name') && isfield(test_imdb, 'name')
    if strcmp(train_imdb.name, test_imdb.name)
      imdb.name = test_imdb.name;
    end
  end
  afprintf(sprintf('[INFO] done!\n'));
