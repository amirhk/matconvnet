% -------------------------------------------------------------------------
function testNetworkExtractionUtils(input_opts)
% -------------------------------------------------------------------------
% Copyright (c) 2017, Amir-Hossein Karimi
% All rights reserved.

% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution

% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.


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
