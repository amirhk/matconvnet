function fh = gaussianUtils()
  % assign function handles so we can call these local functions from elsewhere
  fh.testGaussianUtils = @testGaussianUtils;
  fh.fit2DGaussianAndDrawPositiveSamples = @fit2DGaussianAndDrawPositiveSamples;
  fh.fit2DGaussianAndDrawSamples = @fit2DGaussianAndDrawSamples;
  fh.fit2DGaussianAndDrawSuperSamples = @fit2DGaussianAndDrawSuperSamples;

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
  kernel3 = fit2DGaussianAndDrawSamples(kernel, false);
  kernel4 = fit2DGaussianAndDrawSuperSamples(kernel, false);
  figure;
  subplot(2,4,1), imshow(kernel, []), title('input');
  subplot(2,4,5), mesh(1:1:ndim, 1:1:ndim, kernel);
  subplot(2,4,2), imshow(kernel2, []), title('positive samples');
  subplot(2,4,6), mesh(1:1:ndim, 1:1:ndim, kernel2);
  subplot(2,4,3), imshow(kernel3, []), title('+/- samples');
  subplot(2,4,7), mesh(1:1:ndim, 1:1:ndim, kernel3);
  subplot(2,4,4), imshow(kernel4, []), title('super samples');
  subplot(2,4,8), mesh(1:1:ndim, 1:1:ndim, kernel4);

  disp(min(kernel(:)));
  disp(max(kernel(:)));
  disp(min(kernel2(:)));
  disp(max(kernel2(:)));
  disp(min(kernel3(:)));
  disp(max(kernel3(:)));
  disp(min(kernel4(:)));
  disp(max(kernel4(:)));

% --------------------------------------------------------------------
function sample = fit2DGaussianAndDrawPositiveSamples(kernel, debug_flag)
% --------------------------------------------------------------------
  warning off;
  if nargin < 2
    debug_flag = false;
  end
  dim = size(kernel, 1);
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
  sample = drawPositiveSamplesFrom2DGaussian(mu_y, mu_x, covariance, dim);
  if debug_flag
    fprintf('Done!\n');
    figure; mesh(1:1:dim, 1:1:dim, super_sample);
  end

% --------------------------------------------------------------------
function sample = fit2DGaussianAndDrawSamples(kernel, debug_flag)
% --------------------------------------------------------------------
  dim = size(kernel, 1);
  sample = fit2DGaussianAndDrawPositiveSamples(kernel, debug_flag);
  samplen = sample ./ max(sample(:));
  % sample = sign(randn(dim,dim)) .* samplen;
  thresh = .5;
  a = rand(dim,dim);
  b = a >= thresh;
  c = a < thresh;
  d = b - c; % smaller than thresh gets multiplied by -1
  sample = d .* samplen;
  sample = scaleDrawnSampleToInitialDynamicRange(kernel, sample);

% --------------------------------------------------------------------
function super_sample = fit2DGaussianAndDrawSuperSamples(kernel, debug_flag)
% --------------------------------------------------------------------
  dim = size(kernel, 1);
  sample = fit2DGaussianAndDrawPositiveSamples(kernel, debug_flag);
  samplen = sample ./ max(sample(:));
  super_sample = randn(dim,dim) + sign(randn(dim,dim)) .* samplen;
  super_sample = scaleDrawnSampleToInitialDynamicRange(kernel, super_sample);

  % DEP - OCT 28
  % normalization_factor = max(sample(:));
  % samplen = sample ./ normalization_factor;
  % tmp = randn(dim,dim);
  % super_sample = (tmp / max(abs(tmp(:))) + sign(randn(dim,dim)) .* samplen) * normalization_factor;

  % DEP - Oct 26
  % sample_normalized = sample ./ max(sample(:));
  % super_sample =  ...
  %   sign(randn(dim, dim)) .* ...
  %   (randn(dim, dim) + mixing_factor * sample_normalized);

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
function sample = scaleDrawnSampleToInitialDynamicRange(kernel, sample)
% --------------------------------------------------------------------
  kernel_lower_bound = min(min(min(kernel)), 0); % finally compare with 0 because maybe
  kernel_upper_bound = max(max(max(kernel)), 0); % kernel doesn't have any +ve \ -ve vals
  sample_lower_bound = min(min(min(sample)), 0); % finally compare with 0 because maybe
  sample_upper_bound = max(max(max(sample)), 0); % sample doesn't have any +ve \ -ve vals
  sample = (sample - sample_lower_bound) * ...
    ((kernel_upper_bound - kernel_lower_bound) / (sample_upper_bound - sample_lower_bound)) + ...
    kernel_lower_bound
