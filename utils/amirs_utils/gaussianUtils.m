% -------------------------------------------------------------------------
function fh = gaussianUtils()
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

  % assign function handles so we can call these local functions from elsewhere
  fh.testGaussianUtils = @testGaussianUtils;
  fh.fit1DGaussianAndDrawSamples = @fit1DGaussianAndDrawSamples;
  fh.fit2DGaussianAndDrawSuperSamples = @fit2DGaussianAndDrawSuperSamples;
  fh.fit2DGaussianAndDrawPosNegSamples = @fit2DGaussianAndDrawPosNegSamples;
  fh.fit2DGaussianAndDrawPositiveSamples = @fit2DGaussianAndDrawPositiveSamples;
  fh.fit2DGaussianAndDrawShiftFlipSamples = @fit2DGaussianAndDrawShiftFlipSamples;
  fh.fit2DGaussianAndDrawMultRandnSamples = @fit2DGaussianAndDrawMultRandnSamples;
  fh.fit2DGaussianAndDrawMultKernelSamples = @fit2DGaussianAndDrawMultKernelSamples;

% --------------------------------------------------------------------
function [mu_y, mu_x, covariance] = fit2DGaussian(kernel)
% --------------------------------------------------------------------
  alpha = 1000; % really large number so that smaller weights are also factored
  a = kernel;
  b = abs(a);
  c = b * alpha;
  d = floor(c);
  list = [];
  for y = 1:size(d, 1)
    for x = 1:size(d, 2)
      for count = 1:d(y, x)
        list(end + 1, :) = [y, x];
      end
    end
  end
  if size(list, 1) < 2
    means = list;
  else
    means = mean(list);
  end
  mu_y = means(1);
  mu_x = means(2);
  covariance = cov(list);

% --------------------------------------------------------------------
function averaged_sample_kernel = drawPositiveSamplesFrom2DGaussian( ...
  mu_y, ...
  mu_x, ...
  covariance, ...
  kernel_size)
% --------------------------------------------------------------------
  %2d gaussian
  gaussian2D = @(mu, covariance, X) ...
    1 / (sqrt(det(2 * pi * covariance))) * ...
    exp(-1/2 * (X - mu)' * inv(covariance) * (X - mu));

  mu = [mu_y; mu_x];
  sample_kernel = ones(kernel_size, kernel_size);
  sample_kernels = [];
  trial_repeat_count = 1;
  for k = 1:trial_repeat_count
    for y = 1:kernel_size
      for x = 1:kernel_size
        sample_kernel(y, x) = gaussian2D(mu, covariance, [y; x]);
      end
    end
    sample_kernels(:, :, end + 1) = sample_kernel;
  end

  averaged_sample_kernel = mean(sample_kernels, 3);

% --------------------------------------------------------------------
function testGaussianUtils(kernel)
% --------------------------------------------------------------------
  ndim = size(kernel,1);
  kernel2 = fit2DGaussianAndDrawPositiveSamples(kernel, false);
  kernel3 = fit2DGaussianAndDrawSuperSamples(kernel, false);
  kernel4 = fit2DGaussianAndDrawPosNegSamples(kernel, false);
  kernel5 = fit2DGaussianAndDrawShiftFlipSamples(kernel, false);
  kernel6 = fit2DGaussianAndDrawMultKernelSamples(kernel, false);
  kernel7 = fit2DGaussianAndDrawMultRandnSamples(kernel, false);
  figure;
  i = 1;
  num_kernels = 7;
  subplot(2,num_kernels,i), imshow(kernel, []), title('input');
  subplot(2,num_kernels,num_kernels + i), mesh(1:1:ndim, 1:1:ndim, kernel);
  i = i + 1;
  subplot(2,num_kernels,i), imshow(kernel2, []), title('2D fit / positive samples');
  subplot(2,num_kernels,num_kernels + i), mesh(1:1:ndim, 1:1:ndim, kernel2);
  i = i + 1;
  subplot(2,num_kernels,i), imshow(kernel6, []), title('mult samples');
  subplot(2,num_kernels,num_kernels + i), mesh(1:1:ndim, 1:1:ndim, kernel6);
  i = i + 1;
  subplot(2,num_kernels,i), imshow(kernel7, []), title('multRandn samples');
  subplot(2,num_kernels,num_kernels + i), mesh(1:1:ndim, 1:1:ndim, kernel7);
  i = i + 1;
  subplot(2,num_kernels,i), imshow(kernel3, []), title('super samples');
  subplot(2,num_kernels,num_kernels + i), mesh(1:1:ndim, 1:1:ndim, kernel3);
  i = i + 1;
  subplot(2,num_kernels,i), imshow(kernel4, []), title('posneg samples');
  subplot(2,num_kernels,num_kernels + i), mesh(1:1:ndim, 1:1:ndim, kernel4);
  i = i + 1;
  subplot(2,num_kernels,i), imshow(kernel5, []), title('shiftflip samples');
  subplot(2,num_kernels,num_kernels + i), mesh(1:1:ndim, 1:1:ndim, kernel5);

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                                           -- ==
% -- ==                                       FITTING                                             -- ==
% -- ==                                                                                           -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

% --------------------------------------------------------------------
function sample = fit1DGaussianAndDrawSamples(vector_of_kernels)
% --------------------------------------------------------------------
  dist = fitdist(vector_of_kernels, 'Normal');
  sample = random(dist, [length(vector_of_kernels), 1]);

  % positive_sample = fit2DGaussianAndDrawPositiveSamples(kernel, debug_flag);
  % positive_sample = positive_sample / sum(positive_sample(:));
  % sample = positive_sample .* kernel;

  % % disp(sum(positive_sample(:)));
  % % disp(positive_sample);
  % % g = fspecial('gaussian', [size(kernel,1), size(kernel,1)], 1);
  % % g = fspecial('gaussian', [size(kernel,1), size(kernel,1)], .1);
  % % sample = g .* kernel;
  % % sample2 = g .* kernel;
  % if debug_flag
  %   positive_sample = fit2DGaussianAndDrawPositiveSamples(kernel, debug_flag);
  %   positive_sample = positive_sample / sum(positive_sample(:));
  %   sample1 = positive_sample .* kernel;
  %   g = fspecial('gaussian', [size(kernel,1), size(kernel,1)], 1);
  %   sample2 = g .* kernel;
  %   h = figure;
  %   subplot(1,5,1), imshow(kernel, []), title('input kernel');
  %   subplot(1,5,2), imshow(positive_sample, []), title('positive gaussian');
  %   subplot(1,5,3), imshow(sample1, []), title('positive g * kernel ');
  %   subplot(1,5,4), imshow(g, []), title('fixed gaussian');
  %   subplot(1,5,5), imshow(sample2, []), title('fixed g * kernel');
  %   saveas(h, sprintf('Gaussians - %s.png', datetime('now', 'Format', 'd-MMM-y-HH-mm-ss')));
  % end
  % sample = scaleDrawnSampleToInitialDynamicRange(kernel, sample);
  % % sample = scaleDrawnSampleToInitialDynamicRangeMeanZero(sample);

% --------------------------------------------------------------------
function sample = fit2DGaussianAndDrawMultKernelSamples(kernel, debug_flag)
% --------------------------------------------------------------------
  positive_sample = fit2DGaussianAndDrawPositiveSamples(kernel, debug_flag);
  positive_sample = positive_sample / sum(positive_sample(:));
  sample = positive_sample .* kernel;

  % disp(sum(positive_sample(:)));
  % disp(positive_sample);
  % g = fspecial('gaussian', [size(kernel,1), size(kernel,1)], 1);
  % g = fspecial('gaussian', [size(kernel,1), size(kernel,1)], .1);
  % sample = g .* kernel;
  % sample2 = g .* kernel;
  if debug_flag
    positive_sample = fit2DGaussianAndDrawPositiveSamples(kernel, debug_flag);
    positive_sample = positive_sample / sum(positive_sample(:));
    sample1 = positive_sample .* kernel;
    g = fspecial('gaussian', [size(kernel,1), size(kernel,1)], 1);
    sample2 = g .* kernel;
    h = figure;
    subplot(1,5,1), imshow(kernel, []), title('input kernel');
    subplot(1,5,2), imshow(positive_sample, []), title('positive gaussian');
    subplot(1,5,3), imshow(sample1, []), title('positive g * kernel ');
    subplot(1,5,4), imshow(g, []), title('fixed gaussian');
    subplot(1,5,5), imshow(sample2, []), title('fixed g * kernel');
    saveas(h, sprintf('Gaussians - %s.png', datetime('now', 'Format', 'd-MMM-y-HH-mm-ss')));
  end
  sample = scaleDrawnSampleToInitialDynamicRange(kernel, sample);
  % sample = scaleDrawnSampleToInitialDynamicRangeMeanZero(sample);

% --------------------------------------------------------------------
function sample = fit2DGaussianAndDrawMultRandnSamples(kernel, debug_flag)
% --------------------------------------------------------------------
  ndim = size(kernel, 1);
  positive_sample = fit2DGaussianAndDrawPositiveSamples(kernel, debug_flag);
  sample = randn(ndim, ndim) .* positive_sample;
  sample = scaleDrawnSampleToInitialDynamicRange(kernel, sample);

% --------------------------------------------------------------------
function sample = fit2DGaussianAndDrawSuperSamples(kernel, debug_flag)
% --------------------------------------------------------------------
  ndim = size(kernel, 1);
  sample = fit2DGaussianAndDrawPositiveSamples(kernel, debug_flag);
  samplen = sample ./ max(sample(:));
  sample = randn(ndim,ndim) + sign(randn(ndim,ndim)) .* samplen;
  sample = scaleDrawnSampleToInitialDynamicRange(kernel, sample);

  % DEP - OCT 28
  % normalization_factor = max(sample(:));
  % samplen = sample ./ normalization_factor;
  % tmp = randn(ndim,ndim);
  % super_sample = (tmp / max(abs(tmp(:))) + sign(randn(ndim,ndim)) .* samplen) * normalization_factor;

  % DEP - Oct 26
  % sample_normalized = sample ./ max(sample(:));
  % super_sample =  ...
  %   sign(randn(ndim, ndim)) .* ...
  %   (randn(ndim, ndim) + mixing_factor * sample_normalized);

% --------------------------------------------------------------------
function sample = fit2DGaussianAndDrawPosNegSamples(kernel, debug_flag)
% --------------------------------------------------------------------
  ndim = size(kernel, 1);
  sample = fit2DGaussianAndDrawPositiveSamples(kernel, debug_flag);
  samplen = sample ./ max(sample(:));
  % sample = sign(randn(ndim,ndim)) .* samplen;
  thresh = .5;
  a = rand(ndim,ndim);
  b = a >= thresh;
  c = a < thresh;
  d = b - c; % smaller than thresh gets multiplied by -1
  sample = d .* samplen;
  sample = scaleDrawnSampleToInitialDynamicRange(kernel, sample);

% --------------------------------------------------------------------
function sample = fit2DGaussianAndDrawPositiveSamples(kernel, debug_flag)
% --------------------------------------------------------------------
  warning off;
  if nargin < 2
    debug_flag = false;
  end
  ndim = size(kernel, 1);
  if debug_flag
    fprintf('[INFO] computing params of fitted 2D Gaussian...\n');
  end
  [mu_y, mu_x, covariance] = fit2DGaussian(kernel);
  if debug_flag
    fprintf('[INFO] params of fitted 2D Gaussian, mu_x, mu_y, covariance:\n');
    disp(mu_y);
    disp(mu_x);
    disp(covariance);
    fprintf('[INFO] drawing sample from fitted 2D Gaussian...');
  end
  sample = drawPositiveSamplesFrom2DGaussian(mu_y, mu_x, covariance, ndim);
  if debug_flag
    fprintf('Done!\n');
    figure; mesh(1:1:ndim, 1:1:ndim, sample);
  end

% --------------------------------------------------------------------
function sample = fit2DGaussianAndDrawShiftFlipSamples(kernel, debug_flag)
% --------------------------------------------------------------------
  sample = fit2DGaussianAndDrawPositiveSamples(kernel, debug_flag);
  sample = scaleDrawnSampleToInitialDynamicRange(kernel, sample);
  sample = sample * sign(randn()); % the entire gaussian gets multiplied by +/-, not each pixel

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                                           -- ==
% -- ==                                       SCALING                                             -- ==
% -- ==                                                                                           -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

% --------------------------------------------------------------------
function sample = scaleDrawnSampleToInitialDynamicRange(kernel, sample)
% --------------------------------------------------------------------
  kernel_lower_bound = min(min(min(kernel)), 0); % finally compare with 0 because maybe
  kernel_upper_bound = max(max(max(kernel)), 0); % kernel doesn't have any +ve \ -ve vals
  sample_lower_bound = min(min(min(sample)), 0); % finally compare with 0 because maybe
  sample_upper_bound = max(max(max(sample)), 0); % sample doesn't have any +ve \ -ve vals
  sample = (sample - sample_lower_bound) * ...
    ((kernel_upper_bound - kernel_lower_bound) / (sample_upper_bound - sample_lower_bound)) + ...
    kernel_lower_bound;

% --------------------------------------------------------------------
function sample = scaleDrawnSampleToInitialDynamicRangeBaseZero(kernel, sample)
% --------------------------------------------------------------------
  kernel_lower_bound = min(min(min(kernel)), 0); % finally compare with 0 because maybe
  kernel_upper_bound = max(max(max(kernel)), 0); % kernel doesn't have any +ve \ -ve vals
  sample_lower_bound = min(min(min(sample)), 0); % finally compare with 0 because maybe
  sample_upper_bound = max(max(max(sample)), 0); % sample doesn't have any +ve \ -ve vals
  scaling_factor = ...
    (kernel_upper_bound - kernel_lower_bound) / ...
    (sample_upper_bound - sample_lower_bound);
  sample = sample * scaling_factor;

% --------------------------------------------------------------------
function sample = scaleDrawnSampleToInitialDynamicRangeMeanZero(sample)
% --------------------------------------------------------------------
  sample = sample - mean(sample(:));
