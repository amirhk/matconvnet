function results = testVaryingBatchSizeNetwork(input_opts)
  % -------------------------------------------------------------------------
  %                                                              opts.general
  % -------------------------------------------------------------------------
  opts.general.dataset = getValueFromFieldOrDefault(input_opts, 'dataset', 'mnist-two-class');
  opts.general.network_arch = 'lenet';

  % -------------------------------------------------------------------------
  %                                                                opts.train
  % -------------------------------------------------------------------------
  % opts.train.backprop_depth = getValueFromFieldOrDefault(input_opts, 'backprop_depth', 4);
  % opts.train.batch_size = getValueFromFieldOrDefault(input_opts, 'batch_size', 100);

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
  single_cnn_options.experiment_parent_dir = opts.paths.experiment_dir;
  % single_cnn_options.backprop_depth = opts.train.backprop_depth;
  % single_cnn_options.batch_size = opts.train.batch_size;
  single_cnn_options.debug_flag = false;
  single_cnn_options.gpus = ifNotMacSetGpu(getValueFromFieldOrDefault(opts, 'gpu', 1));

  % all_tests_net = {};
  all_tests_results = {};

  test_number = 1;
  for backprop_depth = [4, 13]
    for batch_size = [100, 256, 512, 1024]
      printConsoleOutputSeparator();
      afprintf(sprintf('Test #%d\n', i));
      single_cnn_options.backprop_depth = backprop_depth;
      single_cnn_options.batch_size = batch_size;
      [~, all_tests_results{i}] = cnnAmir(single_cnn_options);
      results.train_acc(i) = all_tests_results{i}.train.acc;
      results.train_sens(i) = all_tests_results{i}.train.sens;
      results.train_spec(i) = all_tests_results{i}.train.spec;
      results.test_acc(i) = all_tests_results{i}.test.acc;
      results.test_sens(i) = all_tests_results{i}.test.sens;
      results.test_spec(i) = all_tests_results{i}.test.spec;
      test_number = test_number + 1;
    end
  end



  results.train_acc_mean = mean(results.train_acc);
  results.train_sens_mean = mean(results.train_sens);
  results.train_spec_mean = mean(results.train_spec);
  results.train_acc_std = std(results.train_acc);
  results.train_sens_std = std(results.train_sens);
  results.train_spec_std = std(results.train_spec);


  results.test_acc_mean = mean(results.test_acc);
  results.test_sens_mean = mean(results.test_sens);
  results.test_spec_mean = mean(results.test_spec);
  results.test_acc_std = std(results.test_acc);
  results.test_sens_std = std(results.test_sens);
  results.test_spec_std = std(results.test_spec);
  saveStruct2File(results, opts.paths.results_file_path, 0);










% single_cnn_options.dataset = 'svhn';
% single_cnn_options.network_arch = 'lenet';
% single_cnn_options.backprop_depth = 13;
% single_cnn_options.debug_flag = false;
% single_cnn_options.regen = true;
% cnnAmir(single_cnn_options);
