% --------------------------------------------------------------------
function imdb = constructSvhnImdb(opts)
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

  afprintf(sprintf('[INFO] Constructing SVHN imdb...\n'));
  train_file = load(fullfile(opts.imdb.data_dir, 'train_32x32.mat'));
  test_file = load(fullfile(opts.imdb.data_dir, 'test_32x32.mat'));

  data = single(cat(4, train_file.X, test_file.X));
  labels = single(cat(2, train_file.y', test_file.y'));
  set = cat(2, 1 * ones(1, length(train_file.y)), 3 * ones(1, length(test_file.y)));

  % remove mean in any case
  data_mean = mean(data(:,:,:,set == 1), 4);
  data = bsxfun(@minus, data, data_mean);

  % normalize by image mean and std as suggested in `An Analysis of
  % Single-Layer Networks in Unsupervised Feature Learning` Adam
  % Coates, Honglak Lee, Andrew Y. Ng
  if opts.imdb.contrast_normalization
    afprintf(sprintf('[INFO] Contrast-normalizing data... '));
    z = reshape(data,[],size(data, 4));
    z = bsxfun(@minus, z, mean(z,1));
    n = std(z,0,1);
    z = bsxfun(@times, z, mean(n) ./ max(n, 40));
    data = reshape(z, 32, 32, 3, []);
    afprintf(sprintf('done.\n'));
  end

  if opts.imdb.whiten_data
    afprintf(sprintf('[INFO] whitening data... '));
    z = reshape(data,[],size(data, 4));
    W = z(:, set == 1) * z(:, set == 1)' / size(data, 4); % = covariance matrix, scaled by 1 / number_of_train_and_test_images
    [V,D] = eig(W);
    % the scale is selected to approximately preserve the norm of W
    d2 = diag(D);
    en = sqrt(mean(d2));
    whitening_matrix_W = V * diag(en ./ max(sqrt(d2), 10)) * V';
    new_z = whitening_matrix_W * z;
    data = reshape(new_z, 32, 32, 3, []);
    afprintf(sprintf('done.\n'));
  end

  imdb.images.data = data;
  imdb.images.labels = labels;
  imdb.images.set = set;
  imdb.meta.sets = {'train', 'val', 'test'};
  imdb.name = 'norb';

  afprintf(sprintf('done!\n\n'));
  % fh = imdbMultiClassUtils;
  % fh.getImdbInfo(imdb, 1);
  % save(sprintf('%s.mat', imdb.name), 'imdb');
  % afprintf(sprintf('done!\n\n'));

