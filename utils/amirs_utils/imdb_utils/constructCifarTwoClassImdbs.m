% --------------------------------------------------------------------
function [imdb_balanced_high, imdb_balanced_high, imdb_unbalanced] = constructCifarTwoClassImdbs(positive_class_number, negative_class_number)
% --------------------------------------------------------------------
  afprintf(sprintf('[INFO] Constructing unbalanced CIFAR imdbs...\n'));
  opts.imdb.data_dir = fullfile(getDevPath(), 'data', 'source', 'cifar');
  opts.imdb.imdb_portion = 1.0;
  opts.imdb.contrastNormalization = true;
  opts.imdb.whiten_data = true;
  all_cifar_imdb = constructCifarImdb(opts);

  fh_imdb_utils = imdbTwoClassUtils;
  save_file_name_prefix = sprintf( ...
    'saved-two-class-cifar-pos%d-neg%d', ...
    positive_class_number, ...
    negative_class_number);

  % -------------------------------------------------------------------------
  %                                                             balanced-high
  % -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] `balanced-high`...\n'));
  imdb = fh_imdb_utils.constructTwoClassUnbalancedImdb(all_cifar_imdb, positive_class_number, negative_class_number, 1);
  imdb_balanced_high = imdb;
  afprintf(sprintf('done!\n\n'));

  [~] = fh_imdb_utils.getImdbInfo(imdb, true);
  printConsoleOutputSeparator();
  save(sprintf('%s-balanced-train-5000-5000.mat', save_file_name_prefix), 'imdb');

  % -------------------------------------------------------------------------
  %                                                              balanced-low
  % -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] `balanced-low`...\n'));
  imdb = fh_imdb_utils.constructTwoClassUnbalancedImdb(all_cifar_imdb, positive_class_number, negative_class_number, 200);
  imdb = fh_imdb_utils.balanceImdb(imdb, 'train', 'downsample');
  imdb_balanced_high = imdb;
  afprintf(sprintf('done!\n\n'));

  [~] = fh_imdb_utils.getImdbInfo(imdb, true);
  printConsoleOutputSeparator();
  save(sprintf('%s-balanced-train-25-25.mat', save_file_name_prefix), 'imdb');

  % -------------------------------------------------------------------------
  %                                                                unbalanced
  % -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] `unbalanced`...\n'));
  imdb = fh_imdb_utils.constructTwoClassUnbalancedImdb(all_cifar_imdb, positive_class_number, negative_class_number, 200);
  imdb_unbalanced = imdb;
  afprintf(sprintf('done!\n\n'));

  [~] = fh_imdb_utils.getImdbInfo(imdb, true);
  printConsoleOutputSeparator();
  save(sprintf('%s-unbalanced-25-5000.mat', save_file_name_prefix), 'imdb');
