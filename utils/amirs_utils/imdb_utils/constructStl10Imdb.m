% NOTES:
% 1) Using im2double will bring all pixel values between [-1,+1] and hence need
%    higher LR. Note, that constructing CIFAR imdb in matconvnet does not use
%    im2doube by default, but it was recommended by Javad.
% 2) Subtract the mean of the training data from both the training and test data
% 3) STL-10 does NOT require contrast normalization or whitening
% -------------------------------------------------------------------------
function imdb = constructStl10Imdb(opts)
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

  afprintf(sprintf('[INFO] Constructing STL-10 imdb...'));

    train_file = load(fullfile(opts.imdb.data_dir, 'train.mat'));
    test_file = load(fullfile(opts.imdb.data_dir, 'test.mat'));

  data_train = imresize(reshape(im2double(train_file.X'), 96,96,3,[]), [32,32]);
  labels_train = single(train_file.y');
  set_train = 1 * ones(1, 5000);

  data_test = imresize(reshape(im2double(test_file.X'), 96,96,3,[]), [32,32]);
  labels_test = single(test_file.y');
  set_test = 3 * ones(1, 8000);

  data = single(cat(4, data_train, data_test));
  labels = single(cat(2, labels_train, labels_test));
  set = cat(2, set_train, set_test);

  % remove mean in any case
  data_mean = mean(data(:,:,:,set == 1), 4);
  data = bsxfun(@minus, data, data_mean);

  % STL-10 does NOT require contrast normalization or whitening

  imdb.images.data = data;
  imdb.images.labels = labels;
  imdb.images.set = set;
  imdb.meta.sets = {'train', 'val', 'test'};
  imdb.meta.classes = train_file.class_names; % = test_file.class_names
  afprintf(sprintf('done!\n\n'));
