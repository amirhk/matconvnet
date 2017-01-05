function results = testSingleNetwork(input_opts)
  % -------------------------------------------------------------------------
  %                                                              opts.general
  % -------------------------------------------------------------------------
  opts.general.dataset = getValueFromFieldOrDefault(input_opts, 'dataset', 'mnist-two-class');
  switch opts.general.dataset
    case 'prostate'
      opts.general.network_arch = 'prostatenet';
    otherwise % unbalanced mnist, unbalanced cifar, ...
      opts.general.network_arch = 'lenet';
  end

  % -------------------------------------------------------------------------
  %                                                                 opts.imdb
  % -------------------------------------------------------------------------
  opts.imdb.posneg_balance = getValueFromFieldOrDefault(input_opts, 'posneg_balance', 'unbalanced');
  imdb = loadSavedImdb(opts.general.dataset, opts.imdb.posneg_balance);

  % -------------------------------------------------------------------------
  %                                                                opts.train
  % -------------------------------------------------------------------------
  opts.train.backprop_depth = getValueFromFieldOrDefault(input_opts, 'backprop_depth', 4);

  % -------------------------------------------------------------------------
  %                                                                opts.paths
  % -------------------------------------------------------------------------
  opts.paths.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.paths.experiment_parent_dir = getValueFromFieldOrDefault( ...
    input_opts, ...
    'experiment_parent_dir', ...
    fullfile(vl_rootnn, 'experiment_results'));
  opts.paths.experiment_dir = fullfile(opts.paths.experiment_parent_dir, sprintf( ...
    'test-single-network-%s-%s-%s', ...
    opts.paths.time_string, ...
    opts.general.dataset, ...
    opts.general.network_arch));
  if ~exist(opts.paths.experiment_dir)
    mkdir(opts.paths.experiment_dir);
  end
  opts.paths.options_file_path = fullfile(opts.paths.experiment_dir, 'options.txt');
  opts.paths.results_file_path = fullfile(opts.paths.experiment_dir, 'results.txt');

  % -------------------------------------------------------------------------
  %                                                    save experiment setup!
  % -------------------------------------------------------------------------
  saveStruct2File(opts, opts.paths.options_file_path, 0);

  %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

  single_cnn_options.dataset = opts.general.dataset;
  single_cnn_options.network_arch = opts.general.network_arch;
  single_cnn_options.imdb = imdb;
  single_cnn_options.experiment_parent_dir = opts.paths.experiment_dir;
  single_cnn_options.backprop_depth = opts.train.backprop_depth;
  single_cnn_options.weight_decay = 0.0001;
  single_cnn_options.weight_init_source = 'gen';
  single_cnn_options.weight_init_sequence = {'compRand', 'compRand', 'compRand'};
  single_cnn_options.debug_flag = false;
  single_cnn_options.gpus = ifNotMacSetGpu(getValueFromFieldOrDefault(opts, 'gpu', 1));

  all_tests_net = {};
  all_tests_results = {};
  test_repeat_count = 10;

  for i = 1:test_repeat_count
    printConsoleOutputSeparator();
    afprintf(sprintf('Test #%d\n', i));
    [all_tests_net{i}, all_tests_results{i}] = cnnAmir(single_cnn_options);
    results.test_acc(i) = all_tests_results{i}.test.acc;
    results.test_sens(i) = all_tests_results{i}.test.sens;
    results.test_spec(i) = all_tests_results{i}.test.spec;
  end

  results.test_acc_mean = mean(results.test_acc);
  results.test_sens_mean = mean(results.test_sens);
  results.test_spec_mean = mean(results.test_spec);
  results.test_acc_std = std(results.test_acc);
  results.test_sens_std = std(results.test_sens);
  results.test_spec_std = std(results.test_spec);
  saveStruct2File(results, opts.paths.results_file_path, 0);
