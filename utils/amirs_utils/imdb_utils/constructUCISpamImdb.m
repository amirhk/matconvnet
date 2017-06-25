% -------------------------------------------------------------------------
function imdb = constructUCISpamImdb(opts)
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

  afprintf(sprintf('[INFO] Constructing UCI spam imdb...\n'));

  data_file = fullfile(opts.imdb.data_dir, 'spambase.data');
  data_matrix = load(data_file);

  sample_dim = size(data_matrix, 2) - 1;
  number_of_samples = size(data_matrix, 1);
  assert(number_of_samples == 4601);
  number_of_training_samples = 2301;
  number_of_testing_samples = 2300;

  data = data_matrix(:,1:end-1);
  labels = data_matrix(:,end) + 1; % +1 because uci-spam dataset has labels {0,1}, but my code likes {1,2,...}
  set = cat(1, 1 * ones(number_of_training_samples, 1), 3 * ones(number_of_testing_samples, 1));

  assert(length(labels) == length(set));

  % shuffle
  ix = randperm(number_of_samples);
  imdb.images.data = single(data(ix,:));
  imdb.images.labels = single(labels(ix)');
  imdb.images.set = single(set'); % NOT set(ix).... that way you won't have any of your first class samples in the test set!
  imdb.name = 'uci-spam';

  % get the data into 4D format to be compatible with code built for all other imdbs.
  imdb.images.data = reshape(imdb.images.data', sample_dim, 1, 1, []);
  afprintf(sprintf('done!\n\n'));
  fh = imdbMultiClassUtils;
  fh.getImdbInfo(imdb, 1);
  save(sprintf('%s.mat', imdb.name), 'imdb');
