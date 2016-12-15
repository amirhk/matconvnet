num_test_patients = 10;
opts.dataDir = fullfile(getDevPath(), 'matconvnet/data_1/_prostate');
opts.imdbBalancedDir = fullfile(getDevPath(), 'matconvnet/data_1/balanced-prostate-prostatenet');
opts.imdbBalancedPath = fullfile(getDevPath(), 'matconvnet/data_1/balanced-prostate-prostatenet/imdb.mat');
opts.leaveOutType = 'special';
randomPatientIndices = randperm(104);
opts.leaveOutIndices = randomPatientIndices(1:num_test_patients);
opts.contrastNormalization = true;
opts.whitenData = true;
imdb = constructProstateImdb(opts);

images = reshape(imdb.images.data, 3072, [])';
labels = imdb.images.labels;
numExamples = length(labels);
numFeatures = 3072;

%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

Y = labels(1:numExamples);
covtype = images(1:numExamples,1:numFeatures);
tabulate(Y);

%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

is_train = imdb.images.set == 1;
is_test = imdb.images.set == 3;
% part = cvpartition(Y,'holdout',0.5);
% is_train = training(part); % data for fitting
% is_test = test(part); % data for quality assessment

%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

t = templateTree('MinLeafSize',5);
tic
rusTree = fitensemble( ...
  covtype(is_train,:), ...
  Y(is_train), ...
  'RUSBoost', ...
  1000, ...
  t, ...
  'LearnRate', 0.1, ...
  'nprint',20);
toc

%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

l_loss = loss(rusTree, covtype(is_test,:), Y(is_test), 'mode', 'cumulative');

%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

figure;
tic
plot(l_loss);
toc
grid on;
xlabel('Number of trees');
ylabel('Test classification error');

%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

tic
Yfit = predict(rusTree, covtype(is_test,:));
toc
tab = tabulate(Y(is_test));
confusion_matrix = bsxfun(@rdivide, confusionmat(Y(is_test), Yfit), tab(:,2)) * 100;


acc = (1 - l_loss(end)) * 100;
sens = confusion_matrix(1,1);
spec = confusion_matrix(2,2)


fprintf('[INFO] Acc: %6.2f\n', acc);
fprintf('[INFO] Sens: %6.2f\n', sens);
fprintf('[INFO] Spec: %6.2f\n', spec);
