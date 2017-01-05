function runAllTests(dataset, posneg_balance, gpu);

  % -------------------------------------------------------------------------
  %                                                              opts.general
  % -------------------------------------------------------------------------
  opts.general.dataset = dataset;

  % -------------------------------------------------------------------------
  %                                                                 opts.imdb
  % -------------------------------------------------------------------------
  opts.imdb.posneg_balance = posneg_balance;

  % -------------------------------------------------------------------------
  %                                                                opts.train
  % -------------------------------------------------------------------------
  opts.train.gpu = gpu;

  % -------------------------------------------------------------------------
  %                                                                opts.paths
  % -------------------------------------------------------------------------
  opts.paths.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.paths.experiment_parent_dir = getValueFromFieldOrDefault( ...
    {}, ... % TODO: this should be input_opts
    'experiment_parent_dir', ...
    fullfile(vl_rootnn, 'experiment_results'));
  opts.paths.experiment_dir = fullfile(opts.paths.experiment_parent_dir, sprintf( ...
    'test-all-tests-%s-%s-%s-GPU-%d', ...
    opts.paths.time_string, ...
    opts.general.dataset, ...
    opts.imdb.posneg_balance, ...
    opts.train.gpu));
  if ~exist(opts.paths.experiment_dir)
    mkdir(opts.paths.experiment_dir);
  end
  opts.paths.options_file_path = fullfile(opts.paths.experiment_dir, 'options.txt');
  % opts.paths.results_file_path = fullfile(opts.paths.experiment_dir, 'results.txt');

  % -------------------------------------------------------------------------
  %                                                    save experiment setup!
  % -------------------------------------------------------------------------
  saveStruct2File(opts, opts.paths.options_file_path, 0);

  % TODO:
  % ~~~    1. experiment_parent_dir code
  % 2. merge test*.m files (with shared loop function)

  % -------------------------------------------------------------------------
  %                                                            shared options
  % -------------------------------------------------------------------------
  single_test_options.experiment_parent_dir = opts.paths.experiment_dir;
  single_test_options.dataset = opts.general.dataset;
  single_test_options.posneg_balance = opts.imdb.posneg_balance;
  single_test_options.gpu = opts.train.gpu;
  single_test_options.backprop_depth = 4;

  % -------------------------------------------------------------------------
  %                                                               single tree
  % -------------------------------------------------------------------------
  % TODO...

  % -------------------------------------------------------------------------
  %                                                                    forest
  % -------------------------------------------------------------------------
  % % exp 1
  % single_test_options.boosting_method = 'AdaBoostM1';
  % testForest(single_test_options);
  % % exp 2
  % single_test_options.boosting_method = 'RUSBoost';
  % testForest(single_test_options);

  % -------------------------------------------------------------------------
  %                                                                single cnn
  % -------------------------------------------------------------------------
  % exp 1
  single_test_options.backprop_depth = 4;
  testSingleNetwork(single_test_options);
  % exp 2
  single_test_options.backprop_depth = 13;
  testSingleNetwork(single_test_options);

  % % -------------------------------------------------------------------------
  % %                                                              ensemble cnn
  % % -------------------------------------------------------------------------
  % fh = cnnRusboost;
  % % exp 1
  % single_test_options.backprop_depth = 4;
  % single_test_options.symmetric_weight_updates = true;
  % single_test_options.symmetric_loss_updates = true;
  % fh.kFoldCNNRusboost(single_test_options);
  % % exp 2
  % single_test_options.backprop_depth = 4;
  % single_test_options.symmetric_weight_updates = false;
  % single_test_options.symmetric_loss_updates = true;
  % fh.kFoldCNNRusboost(single_test_options);
  % % exp 3
  % single_test_options.backprop_depth = 4;
  % single_test_options.symmetric_weight_updates = true;
  % single_test_options.symmetric_loss_updates = false;
  % fh.kFoldCNNRusboost(single_test_options);
  % % exp 4
  % single_test_options.backprop_depth = 4;
  % single_test_options.symmetric_weight_updates = false;
  % single_test_options.symmetric_loss_updates = false;
  % fh.kFoldCNNRusboost(single_test_options);
  % % exp 5
  % single_test_options.backprop_depth = 13;
  % single_test_options.symmetric_weight_updates = true;
  % single_test_options.symmetric_loss_updates = true;
  % fh.kFoldCNNRusboost(single_test_options);
  % % exp 6
  % single_test_options.backprop_depth = 13;
  % single_test_options.symmetric_weight_updates = false;
  % single_test_options.symmetric_loss_updates = true;
  % fh.kFoldCNNRusboost(single_test_options);
  % % exp 7
  % single_test_options.backprop_depth = 13;
  % single_test_options.symmetric_weight_updates = true;
  % single_test_options.symmetric_loss_updates = false;
  % fh.kFoldCNNRusboost(single_test_options);
  % % exp 8
  % single_test_options.backprop_depth = 13;
  % single_test_options.symmetric_weight_updates = false;
  % single_test_options.symmetric_loss_updates = false;
  % fh.kFoldCNNRusboost(single_test_options);
