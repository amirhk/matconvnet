init_multiplier = 1;
k = 5;



close all
figure
index = 1;
for number_of_examples = [10000] % [10, 100, 1000, 10000, 100000]
  for filter_width = [2] % [1,2,3,4,5]
    m = number_of_examples;
    n = 1;

    gaussian_filter = fspecial('gaussian', [filter_width,filter_width], 1);
    gaussian_random_kernels = init_multiplier * randn(k, k, m, n, 'single');
    smoothed_gaussian_random_kernels = imfilter(gaussian_random_kernels, gaussian_filter);
    vectorized = reshape(smoothed_gaussian_random_kernels, [k*k,m*n])';

    % mean matrix (matricized)
    % tmp = reshape(mean(vectorized), [k, k]);

    % cov matrix
    tmp = cov(vectorized);

    % subplot(5,5,index);
    % imshow(tmp);
    % title(sprintf('num. exmpl.: %d, filter width: %d', number_of_examples, filter_width));
    % index = index + 1;
  end
end

% tmp
figure, imshow(tmp, [])

mu = zeros(25, 1);
sigma = tmp;
r = mvnrnd(mu,sigma,1);

close all
imagesc(reshape(mvnrnd(mu,sigma,1),5,5)); colorbar
colormap gray









% a(:,:,1) = [1,2,3;4,5,6;7,8,9];
% a(:,:,2) = 10 * [1,2,3;4,5,6;7,8,9];
% b = reshape(a, [9,2])';
% cov(b)
