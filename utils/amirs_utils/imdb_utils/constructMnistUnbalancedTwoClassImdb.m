% --------------------------------------------------------------------
function imdb = constructMnistUnbalancedTwoClassImdb(network_arch)
% --------------------------------------------------------------------
  afprintf(sprintf('[INFO] Constructing unbalanced MNIST imdb...\n'));
  opts.imdb.data_dir = fullfile(getDevPath(), 'data', 'source', 'mnist');
  opts.general.network_arch = network_arch;
  imdb = constructMnistImdb(opts);

  fh_imdb_utils = imdbUtils;
  negative_class_number = 9;
  positive_class_number = 1;

  % indices
  train_negative_indices = bsxfun(@and, imdb.images.labels == negative_class_number, imdb.images.set == 1);
  train_positive_indices = bsxfun(@and, imdb.images.labels == positive_class_number, imdb.images.set == 1);
  test_negative_indices = bsxfun(@and, imdb.images.labels == negative_class_number, imdb.images.set == 3);
  test_positive_indices = bsxfun(@and, imdb.images.labels == positive_class_number, imdb.images.set == 3);

  % train set
  data_train_negative = imdb.images.data(:,:,:,train_negative_indices);
  data_train_positive = imdb.images.data(:,:,:,train_positive_indices);
  downsampled_data_train_positive_indices = randsample(size(data_train_positive, 4), floor(size(data_train_positive, 4) / 200));
  downsampled_data_train_positive = data_train_positive(:,:,:, downsampled_data_train_positive_indices);

  data_train_negative = data_train_negative;
  data_train_positive = downsampled_data_train_positive;
  labels_train_negative = 1 * ones(1, size(data_train_negative, 4));
  labels_train_positive = 2 * ones(1, size(data_train_positive, 4));

  data_train = cat(4, data_train_positive, data_train_negative);
  labels_train = cat(2, labels_train_positive, labels_train_negative);

  % shuffle
  ix = randperm(size(data_train, 4));
  data_train = data_train(:,:,:,ix);
  labels_train = labels_train(ix);

  % test set
  data_test_negative = imdb.images.data(:,:,:,test_negative_indices);
  data_test_positive = imdb.images.data(:,:,:,test_positive_indices);
  labels_test_negative = 1 * ones(1, size(data_test_negative, 4));
  labels_test_positive = 2 * ones(1, size(data_test_positive, 4));

  data_test = cat(4, data_test_positive, data_test_negative);
  labels_test = cat(2, labels_test_positive, labels_test_negative);

  % put it all together
  imdb.images.data = cat(4, data_train, data_test);
  imdb.images.labels = cat(2, labels_train, labels_test);
  imdb.images.set = cat(2, 1 * ones(1, size(labels_train, 2)), 3 * ones(1, size(labels_test, 2)));
  % imdb.images.set = (round(rand(1,length(labels_train))) * 2) + 1; % randomly assign to either set 1 or set 3
  imdb.meta.sets = {'train', 'val', 'test'} ;
  imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;
  afprintf(sprintf('done!\n\n'));

  afprintf(sprintf('== == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==\n\n'));
  [~] = fh_imdb_utils.getImdbInfo(imdb, true);
