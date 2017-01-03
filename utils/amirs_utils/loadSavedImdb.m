function imdb = loadSavedImdb(dataset, posneg_balance)
  % dataset: {'mnist-two-class-unbalanced', 'cifar-two-class-unbalanced'}
  % posneg_balance: {'unbalanced', 'balanced-low', 'balanced-high'}
  afprintf(sprintf('[INFO] Loading imdb (dataset: %s, posneg_balance: %s)\n', dataset, posneg_balance));
  switch dataset
    case 'mnist-two-class-unbalanced'
      switch posneg_balance
        case 'unbalanced'
          tmp = load(fullfile(getDevPath(), 'data', 'two_class_imdbs', 'saved-two-class-mnist-pos9-neg4-unbalanced.mat'));
        case 'balanced-low'
          tmp = load(fullfile(getDevPath(), 'data', 'two_class_imdbs', 'saved-two-class-mnist-pos9-neg4-balanced-train-30-30.mat'));
        case 'balanced-high'
          tmp = load(fullfile(getDevPath(), 'data', 'two_class_imdbs', 'saved-two-class-mnist-pos9-neg4-balanced-train-5000-5000.mat'));
      end
    case 'cifar'
      fprintf('TODO: implement!')
  end
  imdb = tmp.imdb;
  afprintf(sprintf('[INFO] done!\n'));

  % print info
  printConsoleOutputSeparator();
  fh_imdb_utils = imdbTwoClassUtils;
  fh_imdb_utils.getImdbInfo(imdb, 1);
  printConsoleOutputSeparator();

