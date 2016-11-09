% -------------------------------------------------------------------------
function imdb = constructSTL10Imdb(opts)
% -------------------------------------------------------------------------
  fprintf('[INFO] Constructing STL-10 imdb...');
  train_file = load(fullfile(opts.dataDir, 'train.mat'));
  test_file = load(fullfile(opts.dataDir, 'test.mat'));

  data_train = imresize(reshape(train_file.X', 96,96,3,[]), [32,32]);
  labels_train = single(train_file.y');
  set_train = 1 * ones(1, 5000);

  data_test = imresize(reshape(test_file.X', 96,96,3,[]), [32,32]);
  labels_test = single(test_file.y');
  set_test = 3 * ones(1, 8000);

  data = single(cat(4, data_train, data_test));
  labels = single(cat(2, labels_train, labels_test));
  set = cat(2, set_train, set_test);

  imdb.images.data = data;
  imdb.images.labels = labels;
  imdb.images.set = set;
  imdb.meta.sets = {'train', 'val', 'test'};
  imdb.meta.classes = train_file.class_names; % = test_file.class_names
  fprintf('done!\n\n');
