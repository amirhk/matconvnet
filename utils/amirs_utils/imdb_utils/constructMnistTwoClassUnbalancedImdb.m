% --------------------------------------------------------------------
function imdb = constructMnistTwoClassUnbalancedImdb(network_arch, negative_class_number, positive_class_number)
% --------------------------------------------------------------------
  afprintf(sprintf('[INFO] Constructing unbalanced MNIST imdb...\n'));
  opts.imdb.data_dir = fullfile(getDevPath(), 'data', 'source', 'mnist');
  opts.general.network_arch = network_arch;
  imdb = constructMnistImdb(opts);

  fh_imdb_utils = imdbTwoClassUtils;
  imdb = fh_imdb_utils.constructTwoClassUnbalancedImdb(imdb, negative_class_number, positive_class_number);
  afprintf(sprintf('done!\n\n'));

  printConsoleOutputSeparator();
  [~] = fh_imdb_utils.getImdbInfo(imdb, true);
  printConsoleOutputSeparator();
