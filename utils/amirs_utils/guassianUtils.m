function fh = guassianUtils()
  % assign function handles so we can call these local functions from elsewhere
  fh.fit2DGaussian = @fit2DGaussian;
  fh.drawSamplesFrom2DGaussian = @drawSamplesFrom2DGaussian;
  fh.drawSuperSamplesFrom2DGaussian = @drawSuperSamplesFrom2DGaussian;

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
function averaged_sample_kernel = drawSamplesFrom2DGaussian( ...
  mu_y, ...
  mu_x, ...
  covariance, ...
  kernel_size)
% --------------------------------------------------------------------
  %2d guassian
  guassian2D = @(mu, covariance, X) ...
    1 / (sqrt(det(2 * pi * covariance))) * ...
    exp(-1/2 * (X - mu)' * inv(covariance) * (X - mu));

  mu = [mu_y; mu_x];
  sample_kernel = ones(kernel_size, kernel_size);
  sample_kernels = [];
  trial_repeat_count = 1;
  for k = 1:trial_repeat_count
    for y = 1:kernel_size
      for x = 1:kernel_size
        sample_kernel(y, x) = guassian2D(mu, covariance, [y; x]);
      end
    end
    sample_kernels(:, :, end + 1) = sample_kernel;
  end

  averaged_sample_kernel = mean(sample_kernels, 3);

% --------------------------------------------------------------------
function super_sample = drawSuperSamplesFrom2DGaussian( ...
  kernel, ...
  mixing_factor, ...
  debug_flag)
% --------------------------------------------------------------------
  if nargin < 2
    mixing_factor = 10;
  end
  if nargin < 3
    debug_flag = false;
  end
  dim = size(kernel, 1);
  if debug_flag
    fprintf('[INFO] computing params of fitted 2D Guassian...\n');
  end
  [mu_y, mu_x, covariance] = fit2DGaussian(kernel);
  if debug_flag
    fprintf('[INFO] params of fitted 2D Guassian, mu_x, mu_y, covariance:\n');
    disp(mu_y);
    disp(mu_x);
    disp(covariance);
    fprintf('[INFO] drawing sample from fitted 2D Guassian...');
  end
  sample = drawSamplesFrom2DGaussian(mu_y, mu_x, covariance, dim);
  if debug_flag
    fprintf('Done!\n');
    fprintf('[INFO] converting sample to "Super" sample...');
  end

  normalization_factor = max(sample(:));
  samplen = sample ./ normalization_factor;
  tmp = randn(dim,dim);
  super_sample = (tmp / max(abs(tmp(:))) + sign(randn(dim,dim)) .* samplen) * normalization_factor;

  % sample_normalized = sample ./ max(sample(:));
  % super_sample =  ...
  %   sign(randn(dim, dim)) .* ...
  %   (randn(dim, dim) + mixing_factor * sample_normalized);

  if debug_flag
    fprintf('Done!\n');
    figure; mesh(1:1:dim, 1:1:dim, super_sample);
  end
