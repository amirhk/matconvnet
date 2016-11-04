a = load('/Users/a6karimi/dev/data/cifar-alexnet/alexnet+8epoch.mat');
% b = load('W1-layer-1.mat');
% b = load('/Users/a6karimi/dev/data/cifar-alexnet/+8epoch-random-from-baseline-2D-mult/W1-layer-1.mat');
b = load('/Volumes/Amir-1/data/cifar-alexnet/+8epoch-random-from-baseline-2D-mult/W1-layer-1.mat');

figure;
kernel = a.net.layers{1}.weights{1}(:,:,1,1);
sample = b.W1(:,:,1,1);
subplot(1,2,1), imshow(kernel, [])
subplot(1,2,2), imshow(sample, [])

utils = gaussianUtils;
d = size(kernel,1);
utils.testGaussianUtils(kernel);






a = load('/Users/a6karimi/dev/data/cifar-alexnet/alexnet+8epoch.mat');
% b = load('W1-layer-3.mat');
% b = load('/Users/a6karimi/dev/data/cifar-alexnet/+8epoch-random-from-baseline-2D-mult/W1-layer-3.mat');
b = load('/Volumes/Amir-1/data/cifar-alexnet/+8epoch-random-from-baseline-2D-mult/W1-layer-3.mat');

figure;
kernel = a.net.layers{3}.weights{1}(:,:,1,1);
sample = b.W1(:,:,1,1);
subplot(1,2,1), imshow(kernel, [])
subplot(1,2,2), imshow(sample, [])

utils = gaussianUtils;
d = size(kernel,1);
utils.testGaussianUtils(kernel);









a = load('/Users/a6karimi/dev/data/cifar-alexnet/alexnet+8epoch.mat');
% b = load('W1-layer-6.mat');
% b = load('/Users/a6karimi/dev/data/cifar-alexnet/+8epoch-random-from-baseline-2D-mult/W1-layer-6.mat');
b = load('/Volumes/Amir-1/data/cifar-alexnet/+8epoch-random-from-baseline-2D-mult/W1-layer-6.mat');

figure;
kernel = a.net.layers{6}.weights{1}(:,:,1,1);
sample = b.W1(:,:,1,1);
subplot(1,2,1), imshow(kernel, [])
subplot(1,2,2), imshow(sample, [])

utils = gaussianUtils;
d = size(kernel,1);
utils.testGaussianUtils(kernel);
