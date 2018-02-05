% --------------------------------------------------------------------
function imdb = constructUSPSImdb(opts)
% --------------------------------------------------------------------
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

  afprintf(sprintf('[INFO] Constructing USPS Fashion imdb...\n'));

  original_data = load(fullfile(opts.imdb.data_dir, 'usps_all.mat'));
  original_data = original_data.data;

  data = zeros(16, 16, 1, 11000);

  filled_counter = 0;
  for j = 1 : size(original_data, 2)
    tmp = original_data(:,j,:);
    tmp = reshape(tmp, [16, 16, 1, 10]);
    data(:,:,:, filled_counter + 1 : filled_counter + 10) = tmp;
    filled_counter = filled_counter + 10;
  end

  total_number_of_samples = size(data, 4)

  number_of_training_samples = .5 * total_number_of_samples;
  number_of_testing_samples = .5 * total_number_of_samples;

  data = data;
  labels = repmat([1:9 0] + 1, [1,1100]);
  set = cat(1, 1 * ones(number_of_training_samples, 1), 3 * ones(number_of_testing_samples, 1));

  assert(length(labels) == length(set));

  % shuffle
  ix = randperm(total_number_of_samples);
  imdb.images.data = data(:, :, :, ix);
  imdb.images.labels = labels(ix);
  imdb.images.set = set; % NOT set(ix).... that way you won't have any of your first class samples in the test set!
  imdb.name = 'usps'

  afprintf(sprintf('done!\n\n'));
  fh = imdbMultiClassUtils;
  fh.getImdbInfo(imdb, 1);
  save(sprintf('%s.mat', imdb.name), 'imdb');
