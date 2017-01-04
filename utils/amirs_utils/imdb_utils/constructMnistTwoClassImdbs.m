% --------------------------------------------------------------------
function [imdb_balanced_high, imdb_balanced_high, imdb_unbalanced] = constructMnistTwoClassImdbs(network_arch, positive_class_number, negative_class_number)
% --------------------------------------------------------------------
  afprintf(sprintf('[INFO] Constructing unbalanced MNIST imdb...\n'));
  opts.imdb.data_dir = fullfile(getDevPath(), 'data', 'source', 'mnist');
  opts.general.network_arch = network_arch;
  all_mnist_imdb = constructMnistImdb(opts);

  fh_imdb_utils = imdbTwoClassUtils;
  save_file_name_prefix = sprintf( ...
    'saved-two-class-mnist-pos%d-neg%d', ...
    positive_class_number, ...
    negative_class_number);

  % -------------------------------------------------------------------------
  %                                                             balanced-high
  % -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] `balanced-high`...\n'));
  imdb = fh_imdb_utils.constructTwoClassUnbalancedImdb(all_mnist_imdb, positive_class_number, negative_class_number, 1);
  imdb_balanced_high = imdb;
  afprintf(sprintf('done!\n\n'));

  [~] = fh_imdb_utils.getImdbInfo(imdb, true);
  printConsoleOutputSeparator();
  save(sprintf('%s-balanced-train-6000-6000.mat', save_file_name_prefix), 'imdb');

  % -------------------------------------------------------------------------
  %                                                              balanced-low
  % -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] `balanced-low`...\n'));
  imdb = fh_imdb_utils.constructTwoClassUnbalancedImdb(all_mnist_imdb, positive_class_number, negative_class_number, 200);
  imdb = fh_imdb_utils.balanceImdb(imdb, 'train', 'downsample');
  imdb_balanced_high = imdb;
  afprintf(sprintf('done!\n\n'));

  [~] = fh_imdb_utils.getImdbInfo(imdb, true);
  printConsoleOutputSeparator();
  save(sprintf('%s-balanced-train-30-30.mat', save_file_name_prefix), 'imdb');

  % -------------------------------------------------------------------------
  %                                                                unbalanced
  % -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] `unbalanced`...\n'));
  imdb = fh_imdb_utils.constructTwoClassUnbalancedImdb(all_mnist_imdb, positive_class_number, negative_class_number, 200);
  imdb_unbalanced = imdb;
  afprintf(sprintf('done!\n\n'));

  [~] = fh_imdb_utils.getImdbInfo(imdb, true);
  printConsoleOutputSeparator();
  save(sprintf('%s-unbalanced-30-6000.mat', save_file_name_prefix), 'imdb');



