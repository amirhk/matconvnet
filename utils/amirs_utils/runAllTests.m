function runAllTests(dataset, posneg_balance, gpus);

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
  opts.train.gpus = gpus;

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
    opts.train.gpus));
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
  % ###    2. merge test*.m files (with shared loop function)

  % -------------------------------------------------------------------------
  %                                                            shared options
  % -------------------------------------------------------------------------
  experiment_options.experiment_parent_dir = opts.paths.experiment_dir;
  experiment_options.dataset = opts.general.dataset;
  experiment_options.posneg_balance = opts.imdb.posneg_balance;
  experiment_options.gpus = opts.train.gpus;

  % -------------------------------------------------------------------------
  %                                                                single svm
  % -------------------------------------------------------------------------
  % experiment_options.training_method = 'svm';
  % % Exp. 1
  % testKFold(experiment_options);

  % -------------------------------------------------------------------------
  %                                                              ensemble svm
  % -------------------------------------------------------------------------
  experiment_options.training_method = 'ensemble-svm';
  experiment_options.iteration_count = 8;
  % Exp. 1
  testKFold(experiment_options);

  % -------------------------------------------------------------------------
  %                                                               single tree
  % -------------------------------------------------------------------------
  % TODO...

  % -------------------------------------------------------------------------
  %                                                                    forest
  % -------------------------------------------------------------------------
  % experiment_options.training_method = 'forest';
  % % Exp. 1
  % experiment_options.boosting_method = 'AdaBoostM1';
  % testKFold(experiment_options);
  % % Exp. 2
  % experiment_options.boosting_method = 'RUSBoost';
  % testKFold(experiment_options);

  % -------------------------------------------------------------------------
  %                                                                single cnn
  % -------------------------------------------------------------------------
  % experiment_options.training_method = 'single-cnn';
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % testKFold(experiment_options);
  % % Exp. 2
  % experiment_options.backprop_depth = 13;
  % testKFold(experiment_options);

  % -------------------------------------------------------------------------
  %                                                              ensemble cnn
  % -------------------------------------------------------------------------
  % experiment_options.training_method = 'ensemble-cnn';
  % experiment_options.iteration_count = 8;
  % % Exp. 1
  % experiment_options.backprop_depth = 4;
  % experiment_options.symmetric_weight_updates = true;
  % experiment_options.symmetric_loss_updates = true;
  % testKFold(experiment_options);
  % % Exp. 2
  % experiment_options.backprop_depth = 4;
  % experiment_options.symmetric_weight_updates = true;
  % experiment_options.symmetric_loss_updates = false;
  % testKFold(experiment_options);
  % % Exp. 3
  % experiment_options.backprop_depth = 4;
  % experiment_options.symmetric_weight_updates = false;
  % experiment_options.symmetric_loss_updates = true;
  % testKFold(experiment_options);
  % % Exp. 4
  % experiment_options.backprop_depth = 4;
  % experiment_options.symmetric_weight_updates = false;
  % experiment_options.symmetric_loss_updates = false;
  % testKFold(experiment_options);
  % % Exp. 5
  % experiment_options.backprop_depth = 13;
  % experiment_options.symmetric_weight_updates = true;
  % experiment_options.symmetric_loss_updates = true;
  % testKFold(experiment_options);
  % % Exp. 6
  % experiment_options.backprop_depth = 13;
  % experiment_options.symmetric_weight_updates = true;
  % experiment_options.symmetric_loss_updates = false;
  % testKFold(experiment_options);
  % % Exp. 7
  % experiment_options.backprop_depth = 13;
  % experiment_options.symmetric_weight_updates = false;
  % experiment_options.symmetric_loss_updates = true;
  % testKFold(experiment_options);
  % % Exp. 8
  % experiment_options.backprop_depth = 13;
  % experiment_options.symmetric_weight_updates = false;
  % experiment_options.symmetric_loss_updates = false;
  % testKFold(experiment_options);
