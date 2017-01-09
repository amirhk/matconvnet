function [trained_model, performance_summary] = testSingleNetwork(input_opts)
  % -------------------------------------------------------------------------
  %                                                              opts.general
  % -------------------------------------------------------------------------
  opts.general.dataset = getValueFromFieldOrDefault(input_opts, 'dataset', 'mnist-two-class-9-4');
  opts.general.network_arch = 'lenet';

  % -------------------------------------------------------------------------
  %                                                                 opts.imdb
  % -------------------------------------------------------------------------
  imdb = getValueFromFieldOrDefault(input_opts, 'imdb', struct());

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

  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  single_cnn_options.dataset = opts.general.dataset;
  single_cnn_options.network_arch = opts.general.network_arch;
  single_cnn_options.imdb = imdb;
  single_cnn_options.experiment_parent_dir = opts.paths.experiment_dir;
  single_cnn_options.backprop_depth = opts.train.backprop_depth;
  single_cnn_options.weight_decay = 0.0001;
  single_cnn_options.weight_init_source = 'gen';
  single_cnn_options.weight_init_sequence = {'compRand', 'compRand', 'compRand'};
  single_cnn_options.debug_flag = false;
  single_cnn_options.gpus = ifNotMacSetGpu(getValueFromFieldOrDefault(input_opts, 'gpus', 1));

  [net, results] = cnnAmir(single_cnn_options);

  trained_model = net;
  performance_summary.weighted_test_accuracy = results.test.acc;
  performance_summary.weighted_test_sensitivity = results.test.sens;
  performance_summary.weighted_test_specificity = results.test.spec;
  saveStruct2File(performance_summary, opts.paths.results_file_path, 0);
