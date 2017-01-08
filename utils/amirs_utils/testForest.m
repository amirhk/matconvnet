function results = testForest(input_opts)

  % -------------------------------------------------------------------------
  %                                                              opts.general
  % -------------------------------------------------------------------------
  opts.general.dataset = getValueFromFieldOrDefault(input_opts, 'dataset', 'mnist-two-class-9-4');

  % -------------------------------------------------------------------------
  %                                                                 opts.imdb
  % -------------------------------------------------------------------------
  imdb = getValueFromFieldOrDefault(input_opts, 'imdb', struct());

  % -------------------------------------------------------------------------
  %                                                                opts.train
  % -------------------------------------------------------------------------
  opts.train.number_of_examples = size(imdb.images.data, 4);
  opts.train.number_of_features = 3072;
  opts.train.number_of_trees = getValueFromFieldOrDefault(input_opts, 'number_of_trees', 1000);
  opts.train.boosting_method = getValueFromFieldOrDefault(input_opts, 'boosting_method', 'RUSBoost'); % {'AdaBoostM1', 'RUSBoost'}

  % -------------------------------------------------------------------------
  %                                                                opts.paths
  % -------------------------------------------------------------------------
  opts.paths.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.paths.experiment_parent_dir = getValueFromFieldOrDefault( ...
    input_opts, ...
    'experiment_parent_dir', ...
    fullfile(vl_rootnn, 'experiment_results'));
  opts.paths.experiment_dir = fullfile(opts.paths.experiment_parent_dir, sprintf( ...
    'test-forest-%s-%s', ...
    opts.paths.time_string, ...
    opts.general.dataset));
  if ~exist(opts.paths.experiment_dir)
    mkdir(opts.paths.experiment_dir);
  end
  opts.paths.options_file_path = fullfile(opts.paths.experiment_dir, 'options.txt');
  opts.paths.results_file_path = fullfile(opts.paths.experiment_dir, 'results.txt');

  % -------------------------------------------------------------------------
  %                                                    save experiment setup!
  % -------------------------------------------------------------------------
  saveStruct2File(opts, opts.paths.options_file_path, 0);

  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  images = reshape(imdb.images.data, 3072, [])';
  labels = imdb.images.labels;
  Y = labels(1:opts.train.number_of_examples);
  cov_type = images(1:opts.train.number_of_examples,1:opts.train.number_of_features);
  is_train = imdb.images.set == 1;
  is_test = imdb.images.set == 3;

  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  printConsoleOutputSeparator();
  afprintf(sprintf('Test #%d\n', i));
  t = templateTree('MinLeafSize',5);
  tic
  rus_tree = fitensemble( ...
    cov_type(is_train,:), ...
    Y(is_train), ...
    opts.train.boosting_method, ...
    opts.train.number_of_trees, ...
    t, ...
    'LearnRate', 0.1, ...
    'nprint', 25);
  toc

  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  l_loss = loss(rus_tree, cov_type(is_test,:), Y(is_test), 'mode', 'cumulative');

  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  % figure;
  % tic
  % plot(l_loss);
  % toc
  % grid on;
  % xlabel('Number of trees');
  % ylabel('Test classification error');

  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  tic
  Yfit = predict(rus_tree, cov_type(is_test,:));
  toc
  % tab = tabulate(Y(is_test));
  % confusion_matrix = bsxfun(@rdivide, confusionmat(Y(is_test), Yfit), tab(:,2)) * 100;
  % acc = (1 - l_loss(end)) * 100;
  % spec = confusion_matrix(1,1);
  % sens = confusion_matrix(2,2);

  labels = Y(is_test);
  predictions = Yfit;
  [ ...
    acc, ...
    sens, ...
    spec, ...
  ] = getAccSensSpec(labels, predictions, true);

  afprintf(sprintf('[INFO] Acc: %.6f\n', acc));
  afprintf(sprintf('[INFO] Sens: %.6f\n', sens));
  afprintf(sprintf('[INFO] Spec: %.6f\n', spec));
  printConsoleOutputSeparator();

  results.weighted_test_accuracy = acc;
  results.weighted_test_sensitivity = sens;
  results.weighted_test_specificity = spec;

