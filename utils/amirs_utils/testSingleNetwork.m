function [all_tests_net, all_tests_results] = testSingleNetwork(dataset, posneg_balance, backprop_depth, gpu)
  all_tests_net = {};
  all_tests_results = {};
  test_repeat_count = 5;
  opts.general.dataset = dataset;
  opts.general.network_arch = 'lenet';
  opts.imdb.posneg_balance = posneg_balance;
  opts.train.backprop_depth = backprop_depth;
  opts.paths.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.paths.experiment_dir = fullfile(vl_rootnn, 'experiment_results', sprintf( ...
    'test-single-network-%s-%s-%s', ...
    opts.general.dataset, ...
    opts.general.network_arch, ...
    opts.paths.time_string));
  if ~exist(opts.paths.experiment_dir)
    mkdir(opts.paths.experiment_dir);
  end
  opts.paths.options_file_path = fullfile(opts.paths.experiment_dir, 'options.txt');
  opts.paths.results_file_path = fullfile(opts.paths.experiment_dir, 'results.txt');
  saveStruct2File(opts, opts.paths.options_file_path, 0);

  single_cnn_options.dataset = opts.general.dataset;
  single_cnn_options.network_arch = opts.general.network_arch;
  single_cnn_options.imdb = loadSavedImdb(dataset, posneg_balance);
  single_cnn_options.experiment_parent_dir = opts.paths.experiment_dir;
  single_cnn_options.backprop_depth = opts.train.backprop_depth;
  single_cnn_options.weight_decay = 0.0001;
  single_cnn_options.weight_init_source = 'gen';
  single_cnn_options.weight_init_sequence = {'compRand', 'compRand', 'compRand'};
  single_cnn_options.debug_flag = false;
  single_cnn_options.regen = true;
  single_cnn_options.gpus = ifNotMacSetGpu(gpu);

  for i = 1:test_repeat_count
    printConsoleOutputSeparator();
    afprintf(sprintf('\nTest #%d\n', i));
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
