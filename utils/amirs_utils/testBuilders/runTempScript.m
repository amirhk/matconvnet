
% imdb = constructSyntheticGaussianImdbNEW(10000, 100, 1, 1);
% getTwoClassCSeparation(imdb)
% getAverageClassEccentricity(imdb)

% imdb = constructSyntheticGaussianImdbNEW(10000, 100, 1000, 1);
% getTwoClassCSeparation(imdb)
% getAverageClassEccentricity(imdb)

% imdb = constructSyntheticGaussianImdbNEW(10000, 100, 1, 1000);
% getTwoClassCSeparation(imdb)
% getAverageClassEccentricity(imdb)

% imdb = constructSyntheticGaussianImdbNEW(10000, 100, 1000, 1000);
% getTwoClassCSeparation(imdb)
% getAverageClassEccentricity(imdb)








% tempScriptReproDasgupta('2_gaussians', 'measure-c-separation');
% tempScriptReproDasgupta('2_gaussians', 'measure-1-knn-perf');
% tempScriptReproDasgupta('2_gaussians', 'measure-eccentricity');
% tempScriptReproDasgupta('2_gaussians', 'measure-linear-svm-perf');
% tempScriptReproDasgupta('2_gaussians', 'measure-mlp-500-100-perf');




% tempScriptReproDasgupta('2_gaussians', 'measure-c-separation', 1, 1);
% tempScriptReproDasgupta('2_gaussians', 'measure-1-knn-perf', 1, 1);

% tempScriptReproDasgupta('2_gaussians', 'measure-c-separation', 0.3, 1);
% tempScriptReproDasgupta('2_gaussians', 'measure-1-knn-perf', 0.3, 1);

% tempScriptReproDasgupta('2_gaussians', 'measure-c-separation', 0.1, 1);
% tempScriptReproDasgupta('2_gaussians', 'measure-1-knn-perf', 0.1, 1);

% tempScriptReproDasgupta('2_gaussians', 'measure-c-separation', 1, 1000);
% tempScriptReproDasgupta('2_gaussians', 'measure-1-knn-perf', 1, 1000);

% tempScriptReproDasgupta('2_gaussians', 'measure-c-separation', 0.3, 1000);
% tempScriptReproDasgupta('2_gaussians', 'measure-1-knn-perf', 0.3, 1000);

% tempScriptReproDasgupta('2_gaussians', 'measure-c-separation', 0.1, 1000);
% tempScriptReproDasgupta('2_gaussians', 'measure-1-knn-perf', 0.1, 1000);




% tempScriptReproDasgupta('5_gaussians', 'measure-c-separation', 1, 1);
% tempScriptReproDasgupta('5_gaussians', 'measure-1-knn-perf', 1, 1);

% tempScriptReproDasgupta('5_gaussians', 'measure-c-separation', 0.3, 1);
% tempScriptReproDasgupta('5_gaussians', 'measure-1-knn-perf', 0.3, 1);

% tempScriptReproDasgupta('5_gaussians', 'measure-c-separation', 0.1, 1);
% tempScriptReproDasgupta('5_gaussians', 'measure-1-knn-perf', 0.1, 1);

% tempScriptReproDasgupta('5_gaussians', 'measure-c-separation', 1, 1000);
% tempScriptReproDasgupta('5_gaussians', 'measure-1-knn-perf', 1, 1000);

% tempScriptReproDasgupta('5_gaussians', 'measure-c-separation', 0.3, 1000);
% tempScriptReproDasgupta('5_gaussians', 'measure-1-knn-perf', 0.3, 1000);

% tempScriptReproDasgupta('5_gaussians', 'measure-c-separation', 0.1, 1000);
% tempScriptReproDasgupta('5_gaussians', 'measure-1-knn-perf', 0.1, 1000);




% tempScriptReproDasgupta('circle_in_ring', 'measure-c-separation', 1, 1);
% tempScriptReproDasgupta('circle_in_ring', 'measure-1-knn-perf', 1, 1);





tempScriptReproDasgupta('cifar-multi-class-subsampled', 'measure-c-separation', 1, 1);
% tempScriptReproDasgupta('cifar-multi-class-subsampled', 'measure-1-knn-perf', 1, 1);

tempScriptReproDasgupta('cifar-no-white-multi-class-subsampled', 'measure-c-separation', 1, 1);
% tempScriptReproDasgupta('cifar-no-white-multi-class-subsampled', 'measure-1-knn-perf', 1, 1);

% tempScriptReproDasgupta('stl-10-multi-class-subsampled', 'measure-c-separation', 1, 1);
% tempScriptReproDasgupta('stl-10-multi-class-subsampled', 'measure-1-knn-perf', 1, 1);

% tempScriptReproDasgupta('mnist-784-two-class-0-1', 'measure-c-separation', 1, 1);
% tempScriptReproDasgupta('mnist-784-two-class-0-1', 'measure-1-knn-perf', 1, 1);

% tempScriptReproDasgupta('mnist-784-two-class-8-3', 'measure-c-separation', 1, 1);
% tempScriptReproDasgupta('mnist-784-two-class-8-3', 'measure-1-knn-perf', 1, 1);

tempScriptReproDasgupta('mnist-784-multi-class-subsampled', 'measure-c-separation', 1, 1);
% tempScriptReproDasgupta('mnist-784-multi-class-subsampled', 'measure-1-knn-perf', 1, 1);

% tempScriptReproDasgupta('svhn-multi-class-subsampled', 'measure-c-separation', 1, 1);
% tempScriptReproDasgupta('svhn-multi-class-subsampled', 'measure-1-knn-perf', 1, 1);

















% functionHandle = @tempScriptRunMmd
% functionHandle = @tempScriptRunTsne;
% functionHandle = @tmpScriptCalculateDistances;
% functionHandle = @tempScriptMeasureClassificationPerformance;
% functionHandle = @tempScriptMeasureCSeparation;
% functionHandle = @tempScriptMeasureAverageClassEccentricity;
% functionHandle = @tempScriptPlot2DEuclideanDistances;
% functionHandle = @tempScriptPlot3DEuclideanDistances;
% functionHandle = @tempScriptPlotProgressionOfRandomProjectionFor1Sample;




% functionHandle('saved-multi-class-gaussian-2D-mean-1-var-0-train-balance-500-test-balance-500', 'balanced-38', 1);
% functionHandle('saved-multi-class-gaussian-2D-mean-1-var-1-train-balance-500-test-balance-500', 'balanced-38', 1);
% functionHandle('saved-multi-class-gaussian-2D-mean-1-var-10-train-balance-500-test-balance-500', 'balanced-38', 1);

% functionHandle('saved-multi-class-gaussian-2D-mean-3-var-3-train-balance-500-test-balance-500', 'balanced-38', 1);

% functionHandle('saved-multi-class-gaussian-3D-mean-1-var-1-diag-train-balance-500-test-balance-500', 'balanced-38', 1);
% functionHandle('saved-multi-class-gaussian-3D-mean-3-var-1-diag-train-balance-500-test-balance-500', 'balanced-38', 1);

% ***
% functionHandle('saved-multi-class-gaussian-3D-mean-3-var-3-train-balance-500-test-balance-500', 'balanced-38', 1);

% functionHandle('saved-multi-class-gaussian-5D-mean-1-var-0-train-balance-500-test-balance-500', 'balanced-38', 1);
% functionHandle('saved-multi-class-gaussian-5D-mean-1-var-1-train-balance-500-test-balance-500', 'balanced-38', 1);
% functionHandle('saved-multi-class-gaussian-5D-mean-1-var-10-train-balance-500-test-balance-500', 'balanced-38', 1);

% functionHandle('saved-multi-class-gaussian-5D-mean-9-var-0-train-balance-500-test-balance-500', 'balanced-38', 1);
% functionHandle('saved-multi-class-gaussian-5D-mean-9-var-1-train-balance-500-test-balance-500', 'balanced-38', 1);
% functionHandle('saved-multi-class-gaussian-5D-mean-9-var-10-train-balance-500-test-balance-500', 'balanced-38', 1);

% functionHandle('saved-multi-class-gaussian-50D-mean-1-var-0-train-balance-500-test-balance-500', 'balanced-38', 1);
% functionHandle('saved-multi-class-gaussian-50D-mean-1-var-1-train-balance-500-test-balance-500', 'balanced-38', 1);
% functionHandle('saved-multi-class-gaussian-50D-mean-1-var-10-train-balance-500-test-balance-500', 'balanced-38', 1);

% functionHandle('saved-multi-class-gaussian-50D-mean-1-var-1-diag-train-balance-500-test-balance-500', 'balanced-38', 1);
% functionHandle('saved-multi-class-gaussian-50D-mean-3-var-1-diag-train-balance-500-test-balance-500', 'balanced-38', 1);
% functionHandle('saved-multi-class-gaussian-50D-mean-3-var-1-diag2-train-balance-500-test-balance-500', 'balanced-38', 1);

% functionHandle('saved-multi-class-gaussian-50D-mean-9-var-0-train-balance-500-test-balance-500', 'balanced-38', 1);
% functionHandle('saved-multi-class-gaussian-50D-mean-9-var-1-train-balance-500-test-balance-500', 'balanced-38', 1);
% functionHandle('saved-multi-class-gaussian-50D-mean-9-var-10-train-balance-500-test-balance-500', 'balanced-38', 1);

% functionHandle('saved-multi-class-gaussian-50D-mean-1-var-10-train-balance-2500-test-balance-2500', 'balanced-38', 1);

% functionHandle('uci-ion', 'balanced-38', 1);
% functionHandle('uci-spam', 'balanced-38', 1);











% functionHandle('mnist-multi-class-subsampled', 'balanced-10', 1);
% functionHandle('mnist-multi-class-subsampled', 'balanced-50', 1);
% functionHandle('mnist-multi-class-subsampled', 'balanced-100', 1);
% functionHandle('mnist-multi-class-subsampled', 'balanced-250', 1);
% functionHandle('mnist-multi-class-subsampled', 'balanced-500', 1);
% functionHandle('mnist-multi-class-subsampled', 'balanced-1000', 1);
% functionHandle('mnist-multi-class-subsampled', 'balanced-2500', 1);
% functionHandle('mnist', 'whatever', 1);


% functionHandle('mnist-784-multi-class-subsampled', 'balanced-10', 1);
% functionHandle('mnist-784-multi-class-subsampled', 'balanced-50', 1);
% functionHandle('mnist-784-multi-class-subsampled', 'balanced-100', 1);
% functionHandle('mnist-784-multi-class-subsampled', 'balanced-250', 1);
% functionHandle('mnist-784-multi-class-subsampled', 'balanced-500', 1);
% functionHandle('mnist-784-multi-class-subsampled', 'balanced-1000', 1);
% functionHandle('mnist-784-multi-class-subsampled', 'balanced-2500', 1);
% not built! functionHandle('mnist-784', 'whatever', 1);




% functionHandle('cifar-multi-class-subsampled', 'balanced-10', 1);
% functionHandle('cifar-multi-class-subsampled', 'balanced-50', 1);
% functionHandle('cifar-multi-class-subsampled', 'balanced-100', 1);
% functionHandle('cifar-multi-class-subsampled', 'balanced-250', 1);
% functionHandle('cifar-multi-class-subsampled', 'balanced-500', 1);
% functionHandle('cifar-multi-class-subsampled', 'balanced-1000', 1);
% functionHandle('cifar-multi-class-subsampled', 'balanced-2500', 1);
% functionHandle('cifar', 'whatever', 1);


% functionHandle('cifar-no-white-multi-class-subsampled', 'balanced-10', 1);
% functionHandle('cifar-no-white-multi-class-subsampled', 'balanced-50', 1);
% functionHandle('cifar-no-white-multi-class-subsampled', 'balanced-100', 1);
% functionHandle('cifar-no-white-multi-class-subsampled', 'balanced-250', 1);
% functionHandle('cifar-no-white-multi-class-subsampled', 'balanced-500', 1);
% functionHandle('cifar-no-white-multi-class-subsampled', 'balanced-1000', 1);
% functionHandle('cifar-no-white-multi-class-subsampled', 'balanced-2500', 1);
% functionHandle('cifar-no-white', 'whatever', 1);




% functionHandle('saved-multi-class-circles-3D-mean-0-var-1-radius-1-train-balance-500-test-balance-500', 'whatever', 1);
% functionHandle('saved-multi-class-circles-50D-mean-0-var-1-radius-1-train-balance-500-test-balance-500', 'whatever', 1);
% functionHandle('saved-multi-class-circles-250D-mean-0-var-1-radius-1-train-balance-500-test-balance-500', 'whatever', 1);
% functionHandle('saved-multi-class-circles-1000D-mean-0-var-1-radius-1-train-balance-500-test-balance-500', 'whatever', 1);
% functionHandle('saved-multi-class-circles-2500D-mean-0-var-1-radius-1-train-balance-500-test-balance-500', 'whatever', 1);




% functionHandle('saved-multi-class-gaussian-3D-mean-1-var-1-train-balance-500-test-balance-500', 'whatever', 1);
% functionHandle('saved-multi-class-gaussian-50D-mean-1-var-1-train-balance-500-test-balance-500', 'whatever', 1);
% functionHandle('saved-multi-class-gaussian-250D-mean-1-var-1-train-balance-500-test-balance-500', 'whatever', 1);
% functionHandle('saved-multi-class-gaussian-1000D-mean-1-var-1-train-balance-500-test-balance-500', 'whatever', 1);
% functionHandle('saved-multi-class-gaussian-2500D-mean-1-var-1-train-balance-500-test-balance-500', 'whatever', 1);

% functionHandle('saved-multi-class-gaussian-3D-mean-1-var-1-train-balance-5000-test-balance-5000', 'whatever', 1);
% functionHandle('saved-multi-class-gaussian-50D-mean-1-var-1-train-balance-5000-test-balance-5000', 'whatever', 1);
% functionHandle('saved-multi-class-gaussian-250D-mean-1-var-1-train-balance-5000-test-balance-5000', 'whatever', 1);
% functionHandle('saved-multi-class-gaussian-1000D-mean-1-var-1-train-balance-5000-test-balance-5000', 'whatever', 1);
% functionHandle('saved-multi-class-gaussian-2500D-mean-1-var-1-train-balance-5000-test-balance-5000', 'whatever', 1);

% functionHandle('saved-multi-class-gaussian-3D-mean-1-var-1-train-balance-50000-test-balance-50000', 'whatever', 1);
% functionHandle('saved-multi-class-gaussian-50D-mean-1-var-1-train-balance-50000-test-balance-50000', 'whatever', 1);
% functionHandle('saved-multi-class-gaussian-250D-mean-1-var-1-train-balance-50000-test-balance-50000', 'whatever', 1);
% functionHandle('saved-multi-class-gaussian-1000D-mean-1-var-1-train-balance-50000-test-balance-50000', 'whatever', 1);
% not built! functionHandle('saved-multi-class-gaussian-2500D-mean-1-var-1-train-balance-50000-test-balance-50000', 'whatever', 1);





% functionHandle('saved-multi-class-gaussian-3D-mean-1-var-10-train-balance-500-test-balance-500', 'whatever', 1);
% functionHandle('saved-multi-class-gaussian-50D-mean-1-var-10-train-balance-500-test-balance-500', 'whatever', 1);
% functionHandle('saved-multi-class-gaussian-250D-mean-1-var-10-train-balance-500-test-balance-500', 'whatever', 1);
% functionHandle('saved-multi-class-gaussian-1000D-mean-1-var-10-train-balance-500-test-balance-500', 'whatever', 1);
% functionHandle('saved-multi-class-gaussian-2500D-mean-1-var-10-train-balance-500-test-balance-500', 'whatever', 1);

% functionHandle('saved-multi-class-spirals-2D-train-balance-500-test-balance-500', 'whatever', 1);

















