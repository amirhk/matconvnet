fh = projectionUtils;




% fh.projectAndSaveImdbThroughNetworkArch('cifar', 'whatever', 'larpV1P1+convV0P0+fcV1', 'compRand', 3);

% fh.projectAndSaveImdbThroughNetworkArch('cifar', 'whatever', 'larpV3P1+convV0P0+fcV1', 'compRand', 7);
fh.projectAndSaveImdbThroughNetworkArch('cifar', 'whatever', 'larpV3P1+convV0P0+fcV1', 'compRandSmoothed', 7);
fh.projectAndSaveImdbThroughNetworkArch('cifar', 'whatever', 'larpV3P1+convV0P0+fcV1', 'bernoulli', 7);
fh.projectAndSaveImdbThroughNetworkArch('cifar', 'whatever', 'larpV3P1+convV0P0+fcV1', 'bernoulliSmoothed', 7);
fh.projectAndSaveImdbThroughNetworkArch('cifar', 'whatever', 'larpV3P1+convV0P0+fcV1', 'gaussian2D', 7);
fh.projectAndSaveImdbThroughNetworkArch('cifar', 'whatever', 'larpV3P1+convV0P0+fcV1', 'gaussian2DMeanSubtracted', 7);
fh.projectAndSaveImdbThroughNetworkArch('cifar', 'whatever', 'larpV3P1+convV0P0+fcV1', 'gaussian2DMeanSubtractedRandomlyFlipped', 7);

% fh.projectAndSaveImdbThroughNetworkArch('cifar', 'whatever', 'larpV3P3+convV0P0+fcV1', 'compRand', 9);























% fh.projectAndSaveImdbThroughNetworkArch('mnist-multi-class-subsampled', 'balanced-38', 'larpV0P0+convV0P0+fcV1', 0);
% fh.projectAndSaveImdbThroughNetworkArch('mnist-multi-class-subsampled', 'balanced-38', 'TESTINGlarpV1P0+convV0P0+fcV1', 2);



% fh.projectAndSaveImdbThroughNetworkArch('mnist-multi-class-subsampled', 'balanced-38', 'larpV1P0ST', 2);
% fh.projectAndSaveImdbThroughNetworkArch('mnist-multi-class-subsampled', 'balanced-38', 'larpV1P1ST', 3);
% fh.projectAndSaveImdbThroughNetworkArch('mnist-multi-class-subsampled', 'balanced-38', 'larpV3P0ST', 6);
% fh.projectAndSaveImdbThroughNetworkArch('mnist-multi-class-subsampled', 'balanced-38', 'larpV3P1ST', 7);
% fh.projectAndSaveImdbThroughNetworkArch('mnist-multi-class-subsampled', 'balanced-38', 'larpV3P3ST', 9);

% fh.projectAndSaveImdbThroughNetworkArch('mnist',                        'whatever',    'larpV1P0ST', 2);
% fh.projectAndSaveImdbThroughNetworkArch('mnist',                        'whatever',    'larpV1P1ST', 3);
% fh.projectAndSaveImdbThroughNetworkArch('mnist',                        'whatever',    'larpV3P0ST', 6);
% fh.projectAndSaveImdbThroughNetworkArch('mnist',                        'whatever',    'larpV3P1ST', 7);
% fh.projectAndSaveImdbThroughNetworkArch('mnist',                        'whatever',    'larpV3P3ST', 9);


% fh.projectAndSaveImdbThroughNetworkArch('cifar-multi-class-subsampled', 'balanced-38', 'larpV3P1+convV0P0+fcV1', 7);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-multi-class-subsampled', 'balanced-38', 'larpV1P0ST', 2);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-multi-class-subsampled', 'balanced-38', 'larpV1P1ST', 3);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-multi-class-subsampled', 'balanced-38', 'larpV3P0ST', 6);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-multi-class-subsampled', 'balanced-38', 'larpV3P1ST', 7);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-multi-class-subsampled', 'balanced-38', 'larpV3P3ST', 9);

% fh.projectAndSaveImdbThroughNetworkArch('cifar',                          'whatever', 'larpV3P1+convV0P0+fcV1', 7);
% fh.projectAndSaveImdbThroughNetworkArch('cifar',                          'whatever', 'larpV3P3+convV0P0+fcV1', 9);
% fh.projectAndSaveImdbThroughNetworkArch('cifar',                        'whatever',    'larpV1P0ST', 2);
% fh.projectAndSaveImdbThroughNetworkArch('cifar',                        'whatever',    'larpV1P1ST', 3);
% fh.projectAndSaveImdbThroughNetworkArch('cifar',                        'whatever',    'larpV3P0ST', 6);
% fh.projectAndSaveImdbThroughNetworkArch('cifar',                        'whatever',    'larpV3P1ST', 7);
% fh.projectAndSaveImdbThroughNetworkArch('cifar',                        'whatever',    'larpV3P3ST', 9);



% fh.projectAndSaveImdbThroughNetworkArch('cifar-no-white-multi-class-subsampled', 'balanced-38', 'larpV1P0ST', 2);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-no-white-multi-class-subsampled', 'balanced-38', 'larpV1P1ST', 3);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-no-white-multi-class-subsampled', 'balanced-38', 'larpV3P0ST', 6);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-no-white-multi-class-subsampled', 'balanced-38', 'larpV3P1ST', 7);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-no-white-multi-class-subsampled', 'balanced-38', 'larpV3P3ST', 9);

% fh.projectAndSaveImdbThroughNetworkArch('cifar-no-white',                        'whatever',    'larpV1P0ST', 2);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-no-white',                        'whatever',    'larpV1P1ST', 3);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-no-white',                        'whatever',    'larpV3P0ST', 6);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-no-white',                        'whatever',    'larpV3P1ST', 7);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-no-white',                        'whatever',    'larpV3P3ST', 9);



% fh.projectAndSaveImdbThroughNetworkArch('coil-100',                        'whatever',    'larpV1P0ST', 2);
% fh.projectAndSaveImdbThroughNetworkArch('coil-100',                        'whatever',    'larpV1P1ST', 3);
% fh.projectAndSaveImdbThroughNetworkArch('coil-100',                        'whatever',    'larpV3P0ST', 6);
% fh.projectAndSaveImdbThroughNetworkArch('coil-100',                        'whatever',    'larpV3P1ST', 7);
% fh.projectAndSaveImdbThroughNetworkArch('coil-100',                        'whatever',    'larpV3P3ST', 9);






















% fh.projectAndSaveImdbThroughNetworkArch('mnist-multi-class-subsampled', 'balanced-38', 'larpV1P0+convV0P0+fcV1', 2);
% fh.projectAndSaveImdbThroughNetworkArch('mnist-multi-class-subsampled', 'balanced-38', 'larpV1P1+convV0P0+fcV1', 3);
% fh.projectAndSaveImdbThroughNetworkArch('mnist-multi-class-subsampled', 'balanced-38', 'larpV3P0+convV0P0+fcV1', 6);
% fh.projectAndSaveImdbThroughNetworkArch('mnist-multi-class-subsampled', 'balanced-38', 'larpV3P1+convV0P0+fcV1', 7);
% fh.projectAndSaveImdbThroughNetworkArch('mnist-multi-class-subsampled', 'balanced-38', 'larpV3P3+convV0P0+fcV1', 9);

% fh.projectAndSaveImdbThroughNetworkArch('mnist',                        'whatever',    'larpV1P0+convV0P0+fcV1', 2);
% fh.projectAndSaveImdbThroughNetworkArch('mnist',                        'whatever',    'larpV1P1+convV0P0+fcV1', 3);
% fh.projectAndSaveImdbThroughNetworkArch('mnist',                        'whatever',    'larpV3P0+convV0P0+fcV1', 6);
% fh.projectAndSaveImdbThroughNetworkArch('mnist',                        'whatever',    'larpV3P1+convV0P0+fcV1', 7);
% fh.projectAndSaveImdbThroughNetworkArch('mnist',                        'whatever',    'larpV3P3+convV0P0+fcV1', 9);



% fh.projectAndSaveImdbThroughNetworkArch('cifar-multi-class-subsampled', 'balanced-38', 'larpV1P0+convV0P0+fcV1', 2);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-multi-class-subsampled', 'balanced-38', 'larpV1P1+convV0P0+fcV1', 3);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-multi-class-subsampled', 'balanced-38', 'larpV3P0+convV0P0+fcV1', 6);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-multi-class-subsampled', 'balanced-38', 'larpV3P1+convV0P0+fcV1', 7);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-multi-class-subsampled', 'balanced-38', 'larpV3P3+convV0P0+fcV1', 9);

% fh.projectAndSaveImdbThroughNetworkArch('cifar',                        'whatever',    'larpV1P0+convV0P0+fcV1', 2);
% fh.projectAndSaveImdbThroughNetworkArch('cifar',                        'whatever',    'larpV1P1+convV0P0+fcV1', 3);
% fh.projectAndSaveImdbThroughNetworkArch('cifar',                        'whatever',    'larpV3P0+convV0P0+fcV1', 6);
% fh.projectAndSaveImdbThroughNetworkArch('cifar',                        'whatever',    'larpV3P1+convV0P0+fcV1', 7);
% fh.projectAndSaveImdbThroughNetworkArch('cifar',                        'whatever',    'larpV3P3+convV0P0+fcV1', 9);



% fh.projectAndSaveImdbThroughNetworkArch('cifar-no-white-multi-class-subsampled', 'balanced-38', 'larpV1P0+convV0P0+fcV1', 2);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-no-white-multi-class-subsampled', 'balanced-38', 'larpV1P1+convV0P0+fcV1', 3);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-no-white-multi-class-subsampled', 'balanced-38', 'larpV3P0+convV0P0+fcV1', 6);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-no-white-multi-class-subsampled', 'balanced-38', 'larpV3P1+convV0P0+fcV1', 7);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-no-white-multi-class-subsampled', 'balanced-38', 'larpV3P3+convV0P0+fcV1', 9);

% fh.projectAndSaveImdbThroughNetworkArch('cifar-no-white',                        'whatever',    'larpV1P0+convV0P0+fcV1', 2);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-no-white',                        'whatever',    'larpV1P1+convV0P0+fcV1', 3);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-no-white',                        'whatever',    'larpV3P0+convV0P0+fcV1', 6);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-no-white',                        'whatever',    'larpV3P1+convV0P0+fcV1', 7);
% fh.projectAndSaveImdbThroughNetworkArch('cifar-no-white',                        'whatever',    'larpV3P3+convV0P0+fcV1', 9);



% fh.projectAndSaveImdbThroughNetworkArch('coil-100',                        'whatever',    'larpV1P0+convV0P0+fcV1', 2);
% fh.projectAndSaveImdbThroughNetworkArch('coil-100',                        'whatever',    'larpV1P1+convV0P0+fcV1', 3);
% fh.projectAndSaveImdbThroughNetworkArch('coil-100',                        'whatever',    'larpV3P0+convV0P0+fcV1', 6);
% fh.projectAndSaveImdbThroughNetworkArch('coil-100',                        'whatever',    'larpV3P1+convV0P0+fcV1', 7);
% fh.projectAndSaveImdbThroughNetworkArch('coil-100',                        'whatever',    'larpV3P3+convV0P0+fcV1', 9);
