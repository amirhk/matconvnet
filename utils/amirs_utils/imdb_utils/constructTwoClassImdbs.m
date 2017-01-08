% --------------------------------------------------------------------
function constructTwoClassImdbs(dataset, network_arch, positive_class_number, negative_class_number)
% --------------------------------------------------------------------
  afprintf(sprintf('[INFO] Constructing unbalanced `%s` imdbs...\n', dataset));
  opts.imdb.data_dir = fullfile(getDevPath(), 'data', 'source', dataset);
  switch dataset
    case 'mnist'
      opts.general.network_arch = network_arch;
      all_class_imdb = constructMnistImdb(opts);
    case 'cifar'
      opts.imdb.imdb_portion = 1.0;
      opts.imdb.contrast_normalization = true;
      opts.imdb.whiten_data = true;
      all_class_imdb = constructCifarImdb(opts);
    case 'svhn'
      opts.imdb.contrast_normalization = true;
      all_class_imdb = constructSvhnImdb(opts);
  end

  fh_imdb_utils = imdbTwoClassUtils;

  % -------------------------------------------------------------------------
  %                                                              balanced-low
  % -------------------------------------------------------------------------
  posneg_balance = 'balanced-low';
  afprintf(sprintf('[INFO] `%s`...\n', posneg_balance));
  imdb = fh_imdb_utils.constructTwoClassUnbalancedImdb(all_class_imdb, positive_class_number, negative_class_number, 200);
  imdb = fh_imdb_utils.balanceImdb(imdb, 'train', 'downsample');
  fh_imdb_utils.saveImdb(imdb, dataset, posneg_balance, positive_class_number, negative_class_number)
  afprintf(sprintf('done!\n\n'));

  % -------------------------------------------------------------------------
  %                                                                unbalanced
  % -------------------------------------------------------------------------
  posneg_balance = 'unbalanced';
  afprintf(sprintf('[INFO] `%s`...\n', posneg_balance));
  imdb = fh_imdb_utils.constructTwoClassUnbalancedImdb(all_class_imdb, positive_class_number, negative_class_number, 200);
  fh_imdb_utils.saveImdb(imdb, dataset, posneg_balance, positive_class_number, negative_class_number)
  afprintf(sprintf('done!\n\n'));

  % -------------------------------------------------------------------------
  %                                                             balanced-high
  % -------------------------------------------------------------------------
  posneg_balance = 'balanced-high';
  afprintf(sprintf('[INFO] `%s`...\n', posneg_balance));
  imdb = fh_imdb_utils.constructTwoClassUnbalancedImdb(all_class_imdb, positive_class_number, negative_class_number, 1);
  fh_imdb_utils.saveImdb(imdb, dataset, posneg_balance, positive_class_number, negative_class_number)
  afprintf(sprintf('done!\n\n'));
