function runAllTests(dataset, posneg_balance);
  opts.dataset = dataset;
  opts.posneg_balance = posneg_balance;
  opts.gpu = 1;
  opts.backprop_depth = 4;

  % single tree
  % TODO...

  % forest
  testForest(opts.dataset, opts.posneg_balance, 'AdaBoostM1');
  testForest(opts.dataset, opts.posneg_balance, 'RUSBoost');

  % single cnn
  testSingleNetwork(opts.dataset, opts.posneg_balance, 4, 1);
  testSingleNetwork(opts.dataset, opts.posneg_balance, 13, 1);

  % ensemble cnn
  fh = cnnRusboost;
  opts.backprop_depth = 4;
  fh.kFoldCNNRusboost(opts);
  opts.backprop_depth = 13;
  fh.kFoldCNNRusboost(opts);
