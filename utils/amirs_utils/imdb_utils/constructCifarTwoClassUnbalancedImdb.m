% --------------------------------------------------------------------
function imdb = constructCifarTwoClassUnbalancedImdb(positive_class_number, negative_class_number)
% --------------------------------------------------------------------
  afprintf(sprintf('[INFO] Constructing unbalanced CIFAR imdb...\n'));
  opts.imdb.data_dir = fullfile(getDevPath(), 'data', 'source', 'cifar');
  opts.imdb.imdb_portion = 1.0;
  opts.imdb.contrastNormalization = true;
  opts.imdb.whiten_data = true;
  imdb = constructCifarImdb(opts);

  fh_imdb_utils = imdbTwoClassUtils;
  imdb = fh_imdb_utils.constructTwoClassUnbalancedImdb(imdb, positive_class_number, negative_class_number);
  afprintf(sprintf('done!\n\n'));

  printConsoleOutputSeparator();
  [~] = fh_imdb_utils.getImdbInfo(imdb, true);
  printConsoleOutputSeparator();
