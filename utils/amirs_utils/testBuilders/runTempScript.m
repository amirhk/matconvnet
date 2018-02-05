% tmp_opts.dataset = 'usps-multi-class-subsampled';
% tmp_opts.posneg_balance = 'balanced-50';
% imdb = loadSavedImdb(tmp_opts, false);
% imdb = getVectorizedImdb(imdb);
% X = imdb.images.data';
% X = X(:,1:100);
% isomap(X)

% function runTempScript()

%   % net = load('/Volumes/Amir/matconvnet/experiment_results/test-classification-perf-2-Aug-2017-10-23-48-rp-tests-cnn-pathology-whatever/simple-CNN-test-accuracy-2-Aug-2017-10-25-16-pathology-whatever-on-convV3P3RL3-RF32CH3+fcV1-RF4CH64-input64x64x3-GPU-3/cnn-2-Aug-2017-10-25-17-cifar-convV3P3RL3-RF32CH3+fcV1-RF4CH64-input64x64x3-batch-size-50-weight-decay-0.0010-GPU-3-bpd-13/net-epoch-100.mat');
%   % net = load('H:\Amir\MATCON~3\EXPERI~1\TEAF3C~1\SIMPLE~1\cnn-2-Aug-2017-10-25-17-cifar-convV3P3RL3-RF32CH3+fcV1-RF4CH64-input64x64x3-batch-size-50-weight-decay-0.0010-GPU-3-bpd-13\net-epoch-100.mat')
%   net = load('H:\Amir\MATCON~3\EXPERI~1\TE3A35~1\SIMPLE~1\cnn-3-Aug-2017-22-03-24-cifar-convV5P3RL5-input64x64x3-batch-size-50-weight-decay-0.0010-GPU-3-bpd-17\net-epoch-36.mat')
%   net = net.net;

%   % tmp_opts.dataset = 'pathology-multi-class-subsampled';
%   % tmp_opts.posneg_balance = 'balanced-50';
%   tmp_opts.dataset = 'pathology';
%   tmp_opts.posneg_balance = 'whatever';
%   imdb = loadSavedImdb(tmp_opts, false);

%   keyboard

%   [top_predictions, ~] = getPredictionsFromModelOnImdb(net, 'cnn', imdb, 3, false);

%   keyboard

%   tmp_imdb = imdb;
%   tmp_imdb.images.data = [];
%   % save('trained_lenet.mat', 'tmp_imdb', 'top_predictions');
%   save('partially_trained_alexnet.mat', 'tmp_imdb', 'top_predictions');
%   % save('trained_alexnet.mat', 'tmp_imdb', 'top_predictions');

%   keyboard

% end



% % -------------------------------------------------------------------------
% function fn = getBatch()
% % -------------------------------------------------------------------------
%   fn = @(x,y) getSimpleNNBatch(x,y);
% end

% % -------------------------------------------------------------------------
% function [images, labels] = getSimpleNNBatch(imdb, batch)
% % -------------------------------------------------------------------------
%   images = imdb.images.data(:,:,:,batch);
%   labels = imdb.images.labels(batch);
%   % if rand > 0.5, images=fliplr(images); end
% end

% % -------------------------------------------------------------------------
% function all_predictions = getAllPredictionsFromTopPredictions(top_predictions, imdb)
% % -------------------------------------------------------------------------
%   % NOT repmat... find the index of the top predicted class
%   % for each sample (in each column) and set that to 1.
%   number_of_samples = size(imdb.images.data, 4);
%   number_of_classes = numel(unique(imdb.images.labels));
%   all_predictions = zeros(number_of_classes, number_of_samples);
%   for i = 1:number_of_samples
%     top_class_prediction = top_predictions(i);
%     all_predictions(top_class_prediction, i) = 1;
%   end
% end















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





% tempScriptReproDasgupta('cifar-multi-class-subsampled', 'measure-c-separation', 1, 1);
% tempScriptReproDasgupta('cifar-multi-class-subsampled', 'measure-1-knn-perf', 1, 1);

% tempScriptReproDasgupta('cifar-no-white-multi-class-subsampled', 'measure-c-separation', 1, 1);
% tempScriptReproDasgupta('cifar-no-white-multi-class-subsampled', 'measure-1-knn-perf', 1, 1);

% tempScriptReproDasgupta('stl-10-multi-class-subsampled', 'measure-c-separation', 1, 1);
% tempScriptReproDasgupta('stl-10-multi-class-subsampled', 'measure-1-knn-perf', 1, 1);

% tempScriptReproDasgupta('mnist-784-two-class-0-1', 'measure-c-separation', 1, 1);
% tempScriptReproDasgupta('mnist-784-two-class-0-1', 'measure-1-knn-perf', 1, 1);

% tempScriptReproDasgupta('mnist-784-two-class-8-3', 'measure-c-separation', 1, 1);
% tempScriptReproDasgupta('mnist-784-two-class-8-3', 'measure-1-knn-perf', 1, 1);

% tempScriptReproDasgupta('mnist-784-multi-class-subsampled', 'measure-c-separation', 1, 1);
% tempScriptReproDasgupta('mnist-784-multi-class-subsampled', 'measure-1-knn-perf', 1, 1);

% tempScriptReproDasgupta('svhn-multi-class-subsampled', 'measure-c-separation', 1, 1);
% tempScriptReproDasgupta('svhn-multi-class-subsampled', 'measure-1-knn-perf', 1, 1);

















% functionHandle = @tempScriptRunMmd
% functionHandle = @tem pScriptRunTsne;
% functionHandle = @tmpScriptCalculateDistances;
% functionHandle = @tmpScriptCalculateDistances2;
functionHandle = @tempScriptMeasureClassificationPerformance;
% functionHandle = @tempScriptMeasureCSeparation;
% functionHandle = @tempScriptMeasureAverageClassEccentricity;
% functionHandle = @tempScriptPlot2DEuclideanDistances;
% functionHandle = @tempScriptPlot3DEuclideanDistances;
% functionHandle = @tempScriptPlotProgressionOfRandomProjectionFor1Sample;





% functionHandle('cifar-multi-class-subsampled', 'balanced-50', '1-knn', 1);
% functionHandle('cifar-multi-class-subsampled', 'balanced-50', 'c-sep', 1);
% functionHandle('cifar-multi-class-subsampled', 'balanced-50', 'cnn', 1);

% functionHandle('cifar-no-white-multi-class-subsampled', 'balanced-50', '1-knn', 1);
% functionHandle('cifar-no-white-multi-class-subsampled', 'balanced-50', 'c-sep', 1);
% functionHandle('cifar-no-white-multi-class-subsampled', 'balanced-50', 'cnn', 1);
% functionHandle('cifar-no-white-multi-class-subsampled', 'balanced-500', '1-knn', 1);
% functionHandle('cifar-no-white-multi-class-subsampled', 'balanced-2500', '1-knn', 1);

% functionHandle('cifar-no-white-two-class-deer-truck', 'balanced-266', '1-knn', 1);

% functionHandle('stl-10-multi-class-subsampled', 'balanced-50', '1-knn', 1);
% functionHandle('stl-10-multi-class-subsampled', 'balanced-50', 'c-sep', 1);
% functionHandle('stl-10-multi-class-subsampled', 'balanced-50', 'cnn', 1);
% functionHandle('stl-10-multi-class-subsampled', 'balanced-500', '1-knn', 1);

% functionHandle('imagenet-tiny-multi-class-subsampled', 'balanced-50', '1-knn', 1);
% functionHandle('imagenet-tiny-multi-class-subsampled', 'balanced-500', '1-knn', 1);

% functionHandle('imagenet-tiny-two-class-brown-bear-german-shepherd', 'balanced-50', '1-knn', 1);
% functionHandle('imagenet-tiny-two-class-brown-bear-german-shepherd', 'balanced-500', '1-knn', 1);

% functionHandle('imagenet-tiny-two-class-school-bus-german-shepherd', 'balanced-50', '1-knn', 1);
% functionHandle('imagenet-tiny-two-class-school-bus-german-shepherd', 'balanced-500', '1-knn', 1);

% functionHandle('mnist-multi-class-subsampled', 'balanced-50', '1-knn', 1);
% functionHandle('mnist-multi-class-subsampled', 'balanced-50', 'c-sep', 1);
% functionHandle('mnist-multi-class-subsampled', 'balanced-50', 'cnn', 1);

% functionHandle('svhn-yes-white-multi-class-subsampled', 'balanced-50', '1-knn', 1);
% functionHandle('svhn-yes-white-multi-class-subsampled', 'balanced-50', 'c-sep', 1);
% functionHandle('svhn-yes-white-multi-class-subsampled', 'balanced-50', 'cnn', 1);

% functionHandle('svhn-multi-class-subsampled', 'balanced-50', '1-knn', 1);
% functionHandle('svhn-multi-class-subsampled', 'balanced-50', 'c-sep', 1);
% functionHandle('svhn-multi-class-subsampled', 'balanced-50', 'cnn', 1);





% -------------------------------------------------------------------------
%                                                                opts.paths
% -------------------------------------------------------------------------
opts.paths.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
opts.paths.experiment_parent_dir = getValueFromFieldOrDefault( ...
  {}, ... % no input_opts here! :)
  'experiment_parent_dir', ...
  fullfile(vl_rootnn, 'experiment_results'));
opts.paths.experiment_dir = fullfile(opts.paths.experiment_parent_dir, sprintf( ...
  'temp-script-%s', ...
  opts.paths.time_string));
if ~exist(opts.paths.experiment_dir)
  mkdir(opts.paths.experiment_dir);
end
% opts.paths.options_file_path = fullfile(opts.paths.experiment_dir, '_options.txt');
opts.paths.results_file_path = fullfile(opts.paths.experiment_dir, '_results.txt');

all_results = {};

% posneg_balance = 'balanced-100';
% dataset_list = { ...
%   'cifar-no-white-two-class-deer-truck', ...
%   ... 'mnist-784-two-class-5-0', ...
% };

posneg_balance = 'balanced-10';
% posneg_balance = 'balanced-1000';
dataset_list = { ...
  'usps-multi-class-subsampled', ...
  ... 'mnist-784-multi-class-subsampled', ...
  ... 'mnist-fashion-multi-class-subsampled', ...
  ... 'shapes-dim-image-32-dim-shape-5-dim-shape-variance-2', ...
  ... 'shapes-dim-image-32-dim-shape-15-dim-shape-variance-5', ...
  ... 'cifar-no-white-multi-class-subsampled', ...
  ... 'stl-10-multi-class-subsampled', ...
  ... 'svhn-multi-class-subsampled', ...
  ... ... 'svhn-no-contrast-multi-class-subsampled', ...
  ... 'norb-96x96x1-multi-class-subsampled', ...
};

% posneg_balance = 'whatever';
% dataset_list = { ...
%   'uci-gisette', ...
%   ... 'uci-ion', ...
%   ... 'uci-spam', ...
%   ... 'mnist-784', ...
%   ... 'svhn', ...
%   ... 'cifar-no-white', ...
%   ... 'coil-100', ...
% };

% posneg_balance = 'balanced-500';
% dataset_list = { ...
%   'imagenet-tiny-two-class-school-bus-remote-control', ...
%   'imagenet-tiny-two-class-school-bus-rocking-chair', ...
%   'imagenet-tiny-two-class-school-bus-steel-arch-bridge', ...
%   'imagenet-tiny-two-class-school-bus-german-shepherd', ...
%   'imagenet-tiny-two-class-monarch-butterfly-lion', ...
%   'imagenet-tiny-two-class-monarch-butterfly-steel-arch-bridge', ...
%   'imagenet-tiny-two-class-lion-brown-bear', ...
%   'imagenet-tiny-two-class-lion-german-shepherd', ...
%   'imagenet-tiny-two-class-brown-bear-german-shepherd', ...
%   'imagenet-tiny-two-class-remote-control-rocking-chair', ...
% };


% posneg_balance = 'balanced-500';
% dataset_list = { ...
%   'mnist-784-two-class-0-1', ...
%   'mnist-784-two-class-0-2', ...
%   'mnist-784-two-class-0-3', ...
%   'mnist-784-two-class-0-4', ...
%   'mnist-784-two-class-5-0', ...
%   'mnist-784-two-class-7-2', ...
%   'mnist-784-two-class-8-2', ...
%   'mnist-784-two-class-8-3', ...
%   'mnist-784-two-class-4-9', ...
%   'mnist-784-two-class-6-9', ...
% };

% posneg_balance = 'balanced-500';
% dataset_list = { ...
%   'svhn-two-class-1-0', ...
%   'svhn-two-class-2-0', ...
%   'svhn-two-class-3-0', ...
%   'svhn-two-class-4-0', ...
%   'svhn-two-class-5-0', ...
%   'svhn-two-class-7-2', ...
%   'svhn-two-class-8-2', ...
%   'svhn-two-class-8-3', ...
%   'svhn-two-class-9-4', ...
%   'svhn-two-class-9-6', ...
% };

% posneg_balance = 'balanced-500';
% dataset_list = { ...
%   'norb-96x96x1-two-class-1-4', ...
%   'norb-96x96x1-two-class-2-3', ...
%   'norb-96x96x1-two-class-2-4', ...
%   'norb-96x96x1-two-class-4-5', ...
%   'norb-96x96x1-two-class-5-3', ...
% };

% posneg_balance = 'balanced-36';
% dataset_list = { ...
%   'coil-100', ...
%   ... 'coil-100-two-class-14-66', ...
%   ... 'coil-100-two-class-19-84', ...
%   ... 'coil-100-two-class-27-26', ...
%   ... 'coil-100-two-class-29-20', ...
%   ... 'coil-100-two-class-32-16', ...
%   ... 'coil-100-two-class-35-31', ...
%   ... 'coil-100-two-class-35-40', ...
%   ... 'coil-100-two-class-38-71', ...
%   ... 'coil-100-two-class-44-34', ...
%   ... 'coil-100-two-class-47-33', ...
%   ... 'coil-100-two-class-47-43', ...
%   ... 'coil-100-two-class-51-47', ...
%   ... 'coil-100-two-class-52-1', ...
%   ... 'coil-100-two-class-55-51', ...
%   ... 'coil-100-two-class-58-57', ...
%   ... 'coil-100-two-class-66-56', ...
%   ... 'coil-100-two-class-72-22', ...
%   ... 'coil-100-two-class-77-34', ...
%   ... 'coil-100-two-class-79-27', ...
%   ... 'coil-100-two-class-82-80', ...
%   ... 'coil-100-two-class-89-22', ...
%   ... 'coil-100-two-class-89-79', ...
%   ... 'coil-100-two-class-93-5', ...
%   ... 'coil-100-two-class-98-35', ...
%   ... 'coil-100-two-class-100-52', ...
% };

dataset_counter = 1;
for dataset_name = dataset_list
  dataset_name = char(dataset_name);
  afprintf(sprintf('[INFO] Testing dataset #%d / %d: %s... \n', dataset_counter, numel(dataset_list), dataset_name));

  all_results{end+1} = functionHandle(dataset_name, posneg_balance, '1-knn', 1, opts.paths.experiment_dir);

  if exist(opts.paths.results_file_path)
    delete(opts.paths.results_file_path);
  end
  saveStruct2File(all_results, opts.paths.results_file_path, 0);

  afprintf(sprintf('[INFO] done!\n'));
  dataset_counter = dataset_counter + 1;
end




% functionHandle('pathology-multi-class-subsampled', 'balanced-50', 'cnn', 1);
% % functionHandle('pathology-multi-class-subsampled', 'balanced-500', 'cnn', 1);
% functionHandle('pathology', 'whatever', 'cnn', 1);
























































































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






% fh_imdb_utils = imdbMultiClassUtils;

% dataset = 'cifar-multi-class-subsampled';
% posneg_balance = 'balanced-38';
% [~, experiments] = setupExperimentsUsingProjectedImbds(dataset, posneg_balance, false, false);
% % fh_imdb_utils.getImdbInfo(experiments{1}.imdb, 1);

% gpu = 1;
% % [best_test_accuracy_mean, best_test_accuracy_std] = getSimpleTestAccuracyFromCnn(dataset, posneg_balance, experiments{1}.imdb, 'convV0P0RL0+fcV1-RF32CH3', gpu);
% % [best_test_accuracy_mean, best_test_accuracy_std] = getSimpleTestAccuracyFromCnn(dataset, posneg_balance, experiments{2}.imdb, 'convV0P0RL0+fcV1-RF16CH64', gpu);
% % [best_test_accuracy_mean, best_test_accuracy_std] = getSimpleTestAccuracyFromCnn(dataset, posneg_balance, experiments{3}.imdb, 'convV0P0RL0+fcV1-RF16CH64', gpu);
% [best_test_accuracy_mean, best_test_accuracy_std] = getSimpleTestAccuracyFromCnn(dataset, posneg_balance, experiments{4}.imdb, 'convV0P0RL0+fcV1-RF16CH64', gpu);


% [best_test_accuracy_mean, best_test_accuracy_std] = getSimpleTestAccuracyFromCnn(dataset, posneg_balance, experiments{2}.imdb, 'convV0P0RL0+fcV1-RF16CH64', gpu);
% [best_test_accuracy_mean, best_test_accuracy_std] = getSimpleTestAccuracyFromCnn(dataset, posneg_balance, experiments{3}.imdb, 'convV0P0RL0+fcV1-RF16CH64', gpu);
% [best_test_accuracy_mean, best_test_accuracy_std] = getSimpleTestAccuracyFromCnn(dataset, posneg_balance, experiments{4}.imdb, 'convV0P0RL0+fcV1-RF16CH64', gpu);


% [best_test_accuracy_mean, best_test_accuracy_std] = getSimpleTestAccuracyFromCnn(dataset, posneg_balance, experiments{2}.imdb, 'convV0P0RL0+fcV1-RF16CH64', gpu);
% [best_test_accuracy_mean, best_test_accuracy_std] = getSimpleTestAccuracyFromCnn(dataset, posneg_balance, experiments{3}.imdb, 'convV0P0RL0+fcV1-RF16CH64', gpu);
% [best_test_accuracy_mean, best_test_accuracy_std] = getSimpleTestAccuracyFromCnn(dataset, posneg_balance, experiments{4}.imdb, 'convV0P0RL0+fcV1-RF16CH64', gpu);









