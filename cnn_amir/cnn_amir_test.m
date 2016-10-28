% for these below, assert a network is returned
cnn_amir_init('networkType', 'alex-net', 'dataset', 'cifar', 'weightInitType', 'compRand', 'weightInitSource', 'load')
cnn_amir_init('networkType', 'alex-net', 'dataset', 'cifar', 'weightInitType', '1D', 'weightInitSource', 'load')
cnn_amir_init('networkType', 'alex-net', 'dataset', 'cifar', 'weightInitType', '2D-super', 'weightInitSource', 'load')
cnn_amir_init('networkType', 'alex-net', 'dataset', 'cifar', 'weightInitType', 'compRand', 'weightInitSource', 'gen')
% DON'T WORK RIGHT NOW
% cnn_amir_init('networkType', 'alex-net', 'dataset', 'cifar', 'weightInitType', '1D', 'weightInitSource', 'gen')
% cnn_amir_init('networkType', 'alex-net', 'dataset', 'cifar', 'weightInitType', '2D-super', 'weightInitSource', 'gen')

% for these below, assert the network trains after only 50 batches of 1 epoch!
cnn_amir('weightInitType', 'compRand', 'weightInitSource', 'load', 'backpropDepth', 20)
cnn_amir('weightInitType', '1D', 'weightInitSource', 'load', 'backpropDepth', 20)
cnn_amir('weightInitType', '2D-super', 'weightInitSource', 'load', 'backpropDepth', 20)'load'
