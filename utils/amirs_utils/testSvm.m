function [trained_model, performance_summary] = testSvm(input_opts)

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

  vectorized_images = reshape(imdb.images.data, 3072, [])';
  labels = imdb.images.labels;
  Y = labels(1:opts.train.number_of_examples);
  is_train = imdb.images.set == 1;
  is_test = imdb.images.set == 3;

  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  printConsoleOutputSeparator();


  svm_struct = svmtrain(vectorized_images(is_train,:), Y(is_train));
  test_predictions = svmclassify(svm_struct , vectorized_images(is_test,:));
  test_labels = Y(is_test);

  [ ...
    acc, ...
    sens, ...
    spec, ...
  ] = getAccSensSpec(test_labels, test_predictions, true);
  printConsoleOutputSeparator();

  trained_model = svm_struct;
  performance_summary.weighted_test_accuracy = acc;
  performance_summary.weighted_test_sensitivity = sens;
  performance_summary.weighted_test_specificity = spec;
  saveStruct2File(performance_summary, opts.paths.results_file_path, 0);
