function [all_tests_net, all_tests_results] = testSingleNetwork(balance_train, backprop_depth, gpu)
  all_tests_net = {};
  all_tests_results = {};
  test_repeat_count = 10;
  opts.general.dataset = 'mnist-two-class-unbalanced';
  opts.general.network_arch = 'lenet';
  opts.imdb.balance_train = balance_train;
  opts.imdb.backprop_depth = backprop_depth;
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

  single_test_options.dataset = opts.general.dataset;
  single_test_options.network_arch = opts.general.network_arch;
  % tmp = load(fullfile(getDevPath(), 'data', 'saved-two-class-mnist-pos9-neg4.mat'));
  tmp = load(fullfile(getDevPath(), 'data', 'saved-two-class-mnist-pos1-neg9.mat'));
  single_test_options.imdb = tmp.imdb;
  single_test_options.experiment_parent_dir = opts.paths.experiment_dir;
  single_test_options.balance_train = opts.imdb.balance_train;
  single_test_options.backprop_depth = opts.imdb.backprop_depth;
  single_test_options.weight_decay = 0.0001;
  single_test_options.weight_init_source = 'gen';
  single_test_options.weight_init_sequence = {'compRand', 'compRand', 'compRand'};
  single_test_options.debug_flag = false;
  single_test_options.regen = true;
  single_test_options.gpus = ifNotMacSetGpu(gpu);

  for i = 1:test_repeat_count
    afprintf(sprintf('\nTest #%d\n', i));
    [all_tests_net{i}, all_tests_results{i}] = cnnAmir(single_test_options);
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
