% --------------------------------------------------------------------
function [imdb_balanced_high, imdb_balanced_low, imdb_unbalanced] = constructTwoClassImdbs(dataset, network_arch, positive_class_number, negative_class_number)
% --------------------------------------------------------------------
  afprintf(sprintf('[INFO] Constructing unbalanced `%s` imdbs...\n', dataset));
  opts.imdb.data_dir = fullfile(getDevPath(), 'data', 'source', dataset);
  switch  da
    case 'mnist'
      opts.general.network_arch = network_arch;
      all_class_imdb = constructMnistImdb(opts);
    case 'cifar'
      opts.imdb.imdb_portion = 1.0;
      opts.imdb.contrastNormalization = true;
      opts.imdb.whiten_data = true;
      all_class_imdb = constructCifarImdb(opts);
    case 'cifar'
      opts.imdb.contrastNormalization = true;
      all_class_imdb = constructSvhnImdb(opts);
  end

  fh_imdb_utils = imdbTwoClassUtils;
  save_file_name_prefix = sprintf( ...
    'saved-two-class-%s-pos%d-neg%d', ...
    dataset,
    positive_class_number, ...
    negative_class_number);

  % -------------------------------------------------------------------------
  %                                                              balanced-low
  % -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] `balanced-low`...\n'));
  imdb = fh_imdb_utils.constructTwoClassUnbalancedImdb(all_class_imdb, positive_class_number, negative_class_number, 200);
  imdb = fh_imdb_utils.balanceImdb(imdb, 'train', 'downsample');
  imdb_balanced_low = imdb;
  afprintf(sprintf('done!\n\n'));

  [~] = fh_imdb_utils.getImdbInfo(imdb, true);
  printConsoleOutputSeparator();
  save(sprintf('%s-balanced-train-30-30.mat', save_file_name_prefix), 'imdb');

  % -------------------------------------------------------------------------
  %                                                                unbalanced
  % -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] `unbalanced`...\n'));
  imdb = fh_imdb_utils.constructTwoClassUnbalancedImdb(all_class_imdb, positive_class_number, negative_class_number, 200);
  imdb_unbalanced = imdb;
  afprintf(sprintf('done!\n\n'));

  [~] = fh_imdb_utils.getImdbInfo(imdb, true);
  printConsoleOutputSeparator();
  save(sprintf('%s-unbalanced-30-6000.mat', save_file_name_prefix), 'imdb');

  % -------------------------------------------------------------------------
  %                                                             balanced-high
  % -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] `balanced-high`...\n'));
  imdb = fh_imdb_utils.constructTwoClassUnbalancedImdb(all_class_imdb, positive_class_number, negative_class_number, 1);
  imdb_balanced_high = imdb;
  afprintf(sprintf('done!\n\n'));

  [~] = fh_imdb_utils.getImdbInfo(imdb, true);
  printConsoleOutputSeparator();
  save(sprintf('%s-balanced-train-6000-6000.mat', save_file_name_prefix), 'imdb');
