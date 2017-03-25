init_multiplier = 1;
k = 5;



% close all
figure
index = 1;

% number_of_examples_list = [10, 100, 1000, 10000, 100000];
% filter_widths = [1,2,3,4,5];

number_of_examples_list = [10000];
filter_widths = [3];

for number_of_examples = number_of_examples_list
  for filter_width = filter_widths
    m = number_of_examples;
    n = 1;

    gaussian_filter = fspecial('gaussian', [filter_width,filter_width], 1);
    % oriented_gaussian_filter =
    eye_filter = eye(filter_width);
    rotated_eye_filter = rot90(eye(filter_width));
    % horiz_filter = zeros()
    prewitt_horiz_filter = fspecial('prewitt');
    prewitt_vert_filter = fspecial('prewitt')';
    sobel_horiz_filter = fspecial('sobel');
    sobel_vert_filter = fspecial('sobel')';


    tmp_filter = gaussian_filter;
    gaussian_random_kernels = init_multiplier * randn(k, k, m, n, 'single');
    filtered_gaussian_random_kernels = imfilter(gaussian_random_kernels, tmp_filter);
    vectorized = reshape(filtered_gaussian_random_kernels, [k*k,m*n])';

    % mean matrix (matricized)
    % tmp = reshape(mean(vectorized), [k, k]);

    % cov matrix
    tmp = cov(vectorized);

    % subplot(numel(number_of_examples_list), numel(filter_widths), index);
    % imshow(tmp, []);
    % title(sprintf('num. exmpl.: %d, filter width: %d', number_of_examples, filter_width));
    % index = index + 1;
  end
end









% tmp
% figure, imshow(tmp, [])

% mu = zeros(25, 1);
% sigma = tmp;
% r = mvnrnd(mu,sigma,1);

close all
figure
colormap gray
for i = 1:100
  imagesc(reshape(mvnrnd(mu,sigma,1),5,5)); colorbar
  pause(0.01);
end










% a(:,:,1) = [1,2,3;4,5,6;7,8,9];
% a(:,:,2) = 10 * [1,2,3;4,5,6;7,8,9];
% b = reshape(a, [9,2])';
% cov(b)
