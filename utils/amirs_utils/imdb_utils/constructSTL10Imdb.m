% NOTES:
% 1) Using im2double will bring all pixel values between [-1,+1] and hence need
%    higher LR. Note, that constructing CIFAR imdb in matconvnet does not use
%    im2doube by default, but it was recommended by Javad.
% 2) Subtract the mean of the training data from both the training and test data
% 3) STL-10 does NOT require contrast normalization or whitening
% -------------------------------------------------------------------------
function imdb = constructSTL10Imdb(opts)
% -------------------------------------------------------------------------
  fprintf('[INFO] Constructing STL-10 imdb...');

  train_file = load(fullfile(opts.imdb.dataDir, 'train.mat'));
  test_file = load(fullfile(opts.imdb.dataDir, 'test.mat'));

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
  dataMean = mean(data(:,:,:,set == 1), 4);
  data = bsxfun(@minus, data, dataMean);

  % STL-10 does NOT require contrast normalization or whitening

  imdb.images.data = data;
  imdb.images.labels = labels;
  imdb.images.set = set;
  imdb.meta.sets = {'train', 'val', 'test'};
  imdb.meta.classes = train_file.class_names; % = test_file.class_names
  fprintf('done!\n\n');
