% -------------------------------------------------------------------------
function [train_imdb, test_imdb] = splitImdb(imdb, debug_flag)
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

  % TODO: assert input is a 4D imdb.

  train_set_indices = imdb.images.set == 1;
  test_set_indices = imdb.images.set == 3;

  data_train = imdb.images.data(:,:,:,train_set_indices);
  data_test = imdb.images.data(:,:,:,test_set_indices);

  labels_train = imdb.images.labels(train_set_indices);
  labels_test = imdb.images.labels(test_set_indices);

  set_train = imdb.images.set(train_set_indices);
  set_test = imdb.images.set(test_set_indices);

  train_imdb = imdb; % in case there is meta information such as imdb.name to be copied over.
  test_imdb = imdb; % in case there is meta information such as imdb.name to be copied over.

  train_imdb.images.data = data_train;
  train_imdb.images.labels = labels_train;
  train_imdb.images.set = set_train;

  test_imdb.images.data = data_test;
  test_imdb.images.labels = labels_test;
  test_imdb.images.set = set_test;
