% load('/Users/a6karimi/dev/forest/forestcover')
% covtype = forestcover;

% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

% numExamples = 5000;
% numFeatures = 10;

% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

% Y = covtype(:,end);
% Y = Y(1:numExamples);
% covtype(:,end) = [];
% covtype = covtype(1:numExamples,1:numFeatures);
% tabulate(Y);

%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

% load('images_and_labels.mat');
% numExamples = 1000;
% numFeatures = 3072;

%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

opts.dataDir = '/Users/a6karimi/dev/matconvnet/data_1/_prostate';
opts.imdbBalancedDir = '/Users/a6karimi/dev/matconvnet/data_1/balanced-prostate-prostatenet';
opts.imdbBalancedPath = '/Users/a6karimi/dev/matconvnet/data_1/balanced-prostate-prostatenet/imdb.mat';
opts.leaveOutType = 'none';
opts.leaveOutIndex = 1;
opts.contrastNormalization = true;
opts.whitenData = true;
imdb = constructProstateImdb(opts);


images = reshape(imdb.images.data, 3072, [])';
labels = imdb.images.labels;
numExamples = length(labels);
% numExamples = 1000;
numFeatures = 3072;

%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

Y = labels(1:numExamples);
covtype = images(1:numExamples,1:numFeatures);
tabulate(Y);

%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

part = cvpartition(Y,'holdout',0.5);
istrain = training(part); % data for fitting
istest = test(part); % data for quality assessment
tabulate(Y(istrain))

%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

t = templateTree('MinLeafSize',5);
tic
rusTree = fitensemble(covtype(istrain,:),Y(istrain),'RUSBoost',1000,t,...
    'LearnRate',0.1,'nprint',20);
toc

%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

l_loss = loss(rusTree,covtype(istest,:),Y(istest),'mode','cumulative');

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
Yfit = predict(rusTree,covtype(istest,:));
toc
tab = tabulate(Y(istest));
bsxfun(@rdivide,confusionmat(Y(istest),Yfit),tab(:,2))*100
disp(1- l_loss(end));

% sum((Yfit == Y(istest)') .* (Yfit == 1)) % TP
% sum((Yfit ~= Y(istest)') .* (Yfit == 1)) % FP
% sum((Yfit == Y(istest)') .* (Yfit == 2)) % TN
% sum((Yfit ~= Y(istest)') .* (Yfit == 2)) % FN


% t = templateTree('MinLeafSize',5);
% tic
% rusTree = fitensemble(covtype(istrain,:),Y(istrain),'AdaBoostM1',1000,t,...
%     'LearnRate',0.1,'nprint',100);
% toc

% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

% l_loss = loss(rusTree,covtype(istest,:),Y(istest),'mode','cumulative');

% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

% figure;
% tic
% plot(l_loss);
% toc
% grid on;
% xlabel('Number of trees');
% ylabel('Test classification error');

% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

% tic
% Yfit = predict(rusTree,covtype(istest,:));
% toc
% tab = tabulate(Y(istest));
% bsxfun(@rdivide,confusionmat(Y(istest),Yfit),tab(:,2))*100
% disp(1- l_loss(end));
