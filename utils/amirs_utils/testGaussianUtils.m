% -------------------------------------------------------------------------
function testGaussianUtils(input_opts)
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


fprintf('[INFO] starting Guassian Utils test...');
utils = gaussianUtils;
% kernel = rand(25,25) - mean(mean(rand(25,25)));
% kernel = randn(25,25);
% kernel = rand(100,100) - mean(mean(rand(100,100)));
% kernel = [...
%   0,1,0,0,0; ...
%   0,1,0,0,0; ...
%   0,1,0,0,1; ...
%   1,1,1,1,1; ...
%   0,0,0,0,0];
% kernel = [...
%   0,0,0,0,0,0,0,0,0,0; ...
%   0,0,0,0,0,0,0,0,0,0; ...
%   0,0,0,0,0,0,0,0,0,0; ...
%   0,0,0,0,0,0,0,0,0,0; ...
%   1,1,1,1,1,1,1,1,1,1; ...
%   1,1,1,1,1,1,1,1,1,1; ...
%   0,0,0,0,0,0,0,0,0,0; ...
%   0,0,0,0,0,0,0,0,0,0; ...
%   0,0,0,0,0,0,0,0,0,0; ...
%   0,0,0,0,0,0,0,0,0,0];
kernel = [...
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0; ...
  0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0; ...
  0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0; ...
  0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0; ...
  0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0; ...
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0; ...
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0; ...
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0; ...
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0; ...
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0; ...
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0; ...
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0; ...
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0; ...
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0; ...
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0; ...
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0; ...
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0; ...
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0; ...
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0; ...
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
% kernel = randn(10,10);
d = size(kernel,1);



% d = size(kernel,1);
% fh = gaussianUtils;
utils.testGaussianUtils(kernel);












% [mu_y, mu_x, covariance] = utils.fit2DGaussian(a);
% sample = utils.drawPositiveSamplesFrom2DGaussian(mu_y, mu_x, covariance, d);
% fprintf('Done!\n');

% normalization_factor = max(sample(:));
% samplen = sample ./ normalization_factor;
% kernel = (randn(d,d) +  ign(randn(d,d)) .* samplen) * normalization_factor;








% samplen = sample ./ normalization_factor;
% kernel = sign(randn(d,d)) .* (randn(d,d)+10*samplen);

% normalization_factor = max(sample(:));
% samplen = sample ./ normalization_factor;
% disp(sample)
% tmp = randn(d,d);
% kernel = (tmp / max(abs(tmp(:))) + sign(randn(d,d)) .* samplen) * normalization_factor;
% disp(kernel)
% figure; mesh(1:1:d, 1:1:d, kernel);


% oneD = load('W1-layer-1.mat');
% twoD = load('/Users/a6karimi/dev/data/cifar-alexnet/+8epoch-random-from-baseline-1D/W1-layer-1.mat');

% t = [];
% for i = 1:96
%   meanOneD = mean(mean(oneD.W1(:,:,1,i)));
%   meanTwoD = mean(mean(twoD.W1(:,:,1,i)));
%   fprintf('mean 1D: %f, mean 2D: %f\n', meanOneD, meanTwoD);
%   t(end + 1) = meanOneD - meanTwoD;
% end

% mean(t) % should be 0!
