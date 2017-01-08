function testVaryingBatchSize(input_opts)
  % -------------------------------------------------------------------------
  %                                                              opts.general
  % -------------------------------------------------------------------------
  opts.general.dataset = getValueFromFieldOrDefault(input_opts, 'dataset', 'mnist');
  opts.general.network_arch = 'lenet';

  % -------------------------------------------------------------------------
  %                                                                opts.paths
  % -------------------------------------------------------------------------
  opts.paths.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.paths.experiment_parent_dir = getValueFromFieldOrDefault( ...
    input_opts, ...
    'experiment_parent_dir', ...
    fullfile(vl_rootnn, 'experiment_results'));
  opts.paths.experiment_dir = fullfile(opts.paths.experiment_parent_dir, sprintf( ...
    'test-varying-batch-size-%s-%s-%s', ...
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
  single_cnn_options.debug_flag = false;
  single_cnn_options.gpus = ifNotMacSetGpu(getValueFromFieldOrDefault(input_opts, 'gpu', 1));

  % all_tests_net = {};
  all_tests_results = {};

  test_number = 1;
  % for backprop_depth = [4, 13]
  for backprop_depth = [13]
    for batch_size = [10, 25, 100, 250, 500, 1000, 10000, 50000]
      i = test_number;
      printConsoleOutputSeparator();
      afprintf(sprintf('Test #%d\n', test_number));
      single_cnn_options.backprop_depth = backprop_depth;
      single_cnn_options.batch_size = batch_size;
      cnnAmir(single_cnn_options);
      test_number = test_number + 1;
    end
  end

% single_cnn_options.dataset = 'svhn';
% single_cnn_options.gpu = 3;
% cnnAmir(single_cnn_options);
