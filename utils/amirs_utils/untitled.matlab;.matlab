init_multiplier = 1;
k = 5;



% close all
figure
index = 1;

number_of_examples_list = [10, 100, 1000, 10000, 100000];
filter_widths = [1,2,3,4,5];

% number_of_examples_list = [10000];
% filter_widths = [3];

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

    subplot(numel(number_of_examples_list), numel(filter_widths), index);
    imshow(tmp, []);
    title(sprintf('num. exmpl.: %d, filter width: %d', number_of_examples, filter_width));
    index = index + 1;
  end
end


% tmp
% figure, imshow(tmp, [])

% mu = zeros(25, 1);
% sigma = tmp;
% r = mvnrnd(mu,sigma,1);



k = 5;
m = 1;
n = 1;
filter_width = 3;
init_multiplier = 1;

large_number = 100000;
gaussian_filter = fspecial('gaussian', [filter_width, filter_width], 1);
gaussian_random_kernels = randn(k, k, large_number, 'single');
smoothed_gaussian_random_kernels = imfilter(gaussian_random_kernels, gaussian_filter);
% note the transpose at the end of the line below!! reshape(..., [large_number, k * k]) is WRONG!
vectorized = reshape(smoothed_gaussian_random_kernels, [k * k, large_number])';

% mean matrix (matricized)
mu = mean(vectorized);

% cov matrix
sigma_smoothed = cov(vectorized);

% for reference
sigma_identity = eye(25);

% setting mu
prewitt_horiz_filter = [2,2,2,2,2; 1,1,1,1,1; 0,0,0,0,0; -1,-1,-1,-1,-1; -2,-2,-2,-2,-2];
mu = reshape(prewitt_horiz_filter, 1, 25);

close all
figure
colormap gray
for i = 1:20
  subplot(2,6,1),  imagesc(prewitt_horiz_filter),                                 title('prewitt horiz filter');
  subplot(2,6,2),  imagesc(reshape(mvnrnd(mu, sigma_identity, 1), 5, 5)),         title('sigma identity var / 1');
  subplot(2,6,3),  imagesc(reshape(mvnrnd(mu, sigma_identity / 10, 1), 5, 5)),    title('sigma identity var / 10');
  subplot(2,6,4),  imagesc(reshape(mvnrnd(mu, sigma_identity / 100, 1), 5, 5)),   title('sigma identity var / 100');
  subplot(2,6,5),  imagesc(reshape(mvnrnd(mu, sigma_identity / 1000, 1), 5, 5)),  title('sigma identity var / 1000');
  subplot(2,6,6),  imagesc(reshape(mvnrnd(mu, sigma_identity / 10000, 1), 5, 5)), title('sigma identity var / 10000');
  subplot(2,6,8),  imagesc(reshape(mvnrnd(mu, sigma_smoothed, 1), 5, 5)),         title('sigma smoothed var / 1');
  subplot(2,6,9),  imagesc(reshape(mvnrnd(mu, sigma_smoothed / 10, 1), 5, 5)),    title('sigma smoothed var / 10');
  subplot(2,6,10), imagesc(reshape(mvnrnd(mu, sigma_smoothed / 100, 1), 5, 5)),   title('sigma smoothed var / 100');
  subplot(2,6,11), imagesc(reshape(mvnrnd(mu, sigma_smoothed / 1000, 1), 5, 5)),  title('sigma smoothed var / 1000');
  subplot(2,6,12), imagesc(reshape(mvnrnd(mu, sigma_smoothed / 10000, 1), 5, 5)), title('sigma smoothed var / 10000');
  pause(0.05);
end









k = 5;
m = 1;
n = 1;
filter_width = 3;
init_multiplier = 1;

large_number = 100000;
gaussian_random_kernels = init_multiplier * randn(k, k, large_number, 'single');

% ------------------------------------------------------------------------------

% confirmed... this runs on every 2D plane separately (on 3D and 4D matrices)
anisodiffed_gaussian_random_kernels_1 = anisodiff2D(gaussian_random_kernels, 1, 1/7, 30, 2);
% note the transpose at the end of the line below!! reshape(..., [large_number, k * k]) is WRONG!
vectorized_1 = reshape(anisodiffed_gaussian_random_kernels_1, [k * k, large_number])';

% mean matrix (matricized)
mu_1 = mean(vectorized_1);

% cov matrix
sigma_anisodiffed_1 = cov(vectorized_1);

% ------------------------------------------------------------------------------

% confirmed... this runs on every 2D plane separately (on 3D and 4D matrices)
anisodiffed_gaussian_random_kernels_2 = anisodiff2D(gaussian_random_kernels, 2, 1/7, 30, 2);
% note the transpose at the end of the line below!! reshape(..., [large_number, k * k]) is WRONG!
vectorized_2 = reshape(anisodiffed_gaussian_random_kernels_2, [k * k, large_number])';

% mean matrix (matricized)
mu_2 = mean(vectorized_2);

% cov matrix
sigma_anisodiffed_2 = cov(vectorized_2);

% ------------------------------------------------------------------------------

% confirmed... this runs on every 2D plane separately (on 3D and 4D matrices)
anisodiffed_gaussian_random_kernels_3 = anisodiff2D(gaussian_random_kernels, 3, 1/7, 30, 2);
% note the transpose at the end of the line below!! reshape(..., [large_number, k * k]) is WRONG!
vectorized_3 = reshape(anisodiffed_gaussian_random_kernels_3, [k * k, large_number])';

% mean matrix (matricized)
mu_3 = mean(vectorized_3);

% cov matrix
sigma_anisodiffed_3 = cov(vectorized_3);

% ------------------------------------------------------------------------------

% confirmed... this runs on every 2D plane separately (on 3D and 4D matrices)
anisodiffed_gaussian_random_kernels_4 = anisodiff2D(gaussian_random_kernels, 4, 1/7, 30, 2);
% note the transpose at the end of the line below!! reshape(..., [large_number, k * k]) is WRONG!
vectorized_4 = reshape(anisodiffed_gaussian_random_kernels_4, [k * k, large_number])';

% mean matrix (matricized)
mu_4 = mean(vectorized_4);

% cov matrix
sigma_anisodiffed_4 = cov(vectorized_4);

% ------------------------------------------------------------------------------

% for reference
sigma_identity = eye(25);

% setting mu
zero_filter = zeros(5,5);
mu = reshape(zero_filter, 1, 25);

close all
figure
colormap gray
for i = 1:2
  subplot(5, 6, 1 + 6 * (1 - 1)), imagesc(sigma_identity),                                              title('sigma idenity');
  subplot(5, 6, 2 + 6 * (1 - 1)), imagesc(reshape(mvnrnd(mu, sigma_identity, 1), 5, 5)),                title('sigma identity var / 1');
  subplot(5, 6, 3 + 6 * (1 - 1)), imagesc(reshape(mvnrnd(mu, sigma_identity / 10, 1), 5, 5)),           title('sigma identity var / 10');
  subplot(5, 6, 4 + 6 * (1 - 1)), imagesc(reshape(mvnrnd(mu, sigma_identity / 100, 1), 5, 5)),          title('sigma identity var / 100');
  subplot(5, 6, 5 + 6 * (1 - 1)), imagesc(reshape(mvnrnd(mu, sigma_identity / 1000, 1), 5, 5)),         title('sigma identity var / 1000');
  subplot(5, 6, 6 + 6 * (1 - 1)), imagesc(reshape(mvnrnd(mu, sigma_identity / 10000, 1), 5, 5)),        title('sigma identity var / 10000');

  subplot(5, 6, 1 + 6 * (2 - 1)), imagesc(sigma_anisodiffed_1),                                         title('sigma anisodiffed 1');
  subplot(5, 6, 2 + 6 * (2 - 1)), imagesc(reshape(mvnrnd(mu_1, sigma_anisodiffed_1, 1), 5, 5)),         title('sigma smoothed var / 1');
  subplot(5, 6, 3 + 6 * (2 - 1)), imagesc(reshape(mvnrnd(mu_1, sigma_anisodiffed_1 / 10, 1), 5, 5)),    title('sigma smoothed var / 10');
  subplot(5, 6, 4 + 6 * (2 - 1)), imagesc(reshape(mvnrnd(mu_1, sigma_anisodiffed_1 / 100, 1), 5, 5)),   title('sigma smoothed var / 100');
  subplot(5, 6, 5 + 6 * (2 - 1)), imagesc(reshape(mvnrnd(mu_1, sigma_anisodiffed_1 / 1000, 1), 5, 5)),  title('sigma smoothed var / 1000');
  subplot(5, 6, 6 + 6 * (2 - 1)), imagesc(reshape(mvnrnd(mu_1, sigma_anisodiffed_1 / 10000, 1), 5, 5)), title('sigma smoothed var / 10000');

  subplot(5, 6, 1 + 6 * (3 - 1)), imagesc(sigma_anisodiffed_2),                                         title('sigma anisodiffed 2');
  subplot(5, 6, 2 + 6 * (3 - 1)), imagesc(reshape(mvnrnd(mu_2, sigma_anisodiffed_2, 1), 5, 5)),         title('sigma smoothed var / 1');
  subplot(5, 6, 3 + 6 * (3 - 1)), imagesc(reshape(mvnrnd(mu_2, sigma_anisodiffed_2 / 10, 1), 5, 5)),    title('sigma smoothed var / 10');
  subplot(5, 6, 4 + 6 * (3 - 1)), imagesc(reshape(mvnrnd(mu_2, sigma_anisodiffed_2 / 100, 1), 5, 5)),   title('sigma smoothed var / 100');
  subplot(5, 6, 5 + 6 * (3 - 1)), imagesc(reshape(mvnrnd(mu_2, sigma_anisodiffed_2 / 1000, 1), 5, 5)),  title('sigma smoothed var / 1000');
  subplot(5, 6, 6 + 6 * (3 - 1)), imagesc(reshape(mvnrnd(mu_2, sigma_anisodiffed_2 / 10000, 1), 5, 5)), title('sigma smoothed var / 10000');

  subplot(5, 6, 1 + 6 * (4 - 1)), imagesc(sigma_anisodiffed_3),                                         title('sigma anisodiffed 3');
  subplot(5, 6, 2 + 6 * (4 - 1)), imagesc(reshape(mvnrnd(mu_3, sigma_anisodiffed_3, 1), 5, 5)),         title('sigma smoothed var / 1');
  subplot(5, 6, 3 + 6 * (4 - 1)), imagesc(reshape(mvnrnd(mu_3, sigma_anisodiffed_3 / 10, 1), 5, 5)),    title('sigma smoothed var / 10');
  subplot(5, 6, 4 + 6 * (4 - 1)), imagesc(reshape(mvnrnd(mu_3, sigma_anisodiffed_3 / 100, 1), 5, 5)),   title('sigma smoothed var / 100');
  subplot(5, 6, 5 + 6 * (4 - 1)), imagesc(reshape(mvnrnd(mu_3, sigma_anisodiffed_3 / 1000, 1), 5, 5)),  title('sigma smoothed var / 1000');
  subplot(5, 6, 6 + 6 * (4 - 1)), imagesc(reshape(mvnrnd(mu_3, sigma_anisodiffed_3 / 10000, 1), 5, 5)), title('sigma smoothed var / 10000');

  subplot(5, 6, 1 + 6 * (5 - 1)), imagesc(sigma_anisodiffed_4),                                         title('sigma anisodiffed 4');
  subplot(5, 6, 2 + 6 * (5 - 1)), imagesc(reshape(mvnrnd(mu_4, sigma_anisodiffed_4, 1), 5, 5)),         title('sigma smoothed var / 1');
  subplot(5, 6, 3 + 6 * (5 - 1)), imagesc(reshape(mvnrnd(mu_4, sigma_anisodiffed_4 / 10, 1), 5, 5)),    title('sigma smoothed var / 10');
  subplot(5, 6, 4 + 6 * (5 - 1)), imagesc(reshape(mvnrnd(mu_4, sigma_anisodiffed_4 / 100, 1), 5, 5)),   title('sigma smoothed var / 100');
  subplot(5, 6, 5 + 6 * (5 - 1)), imagesc(reshape(mvnrnd(mu_4, sigma_anisodiffed_4 / 1000, 1), 5, 5)),  title('sigma smoothed var / 1000');
  subplot(5, 6, 6 + 6 * (5 - 1)), imagesc(reshape(mvnrnd(mu_4, sigma_anisodiffed_4 / 10000, 1), 5, 5)), title('sigma smoothed var / 10000');
  pause(0.05);
end









k = 5;
m = 1;
n = 1;
filter_width = 3;
init_multiplier = 1;

large_number = 100000;
gaussian_random_kernels = init_multiplier * randn(k, k, large_number, 'single');

% ------------------------------------------------------------------------------

filter_width = 2;
gaussian_filter = fspecial('gaussian', [filter_width,filter_width], 1);
smoothed_gaussian_random_kernels_1 = imfilter(gaussian_random_kernels, gaussian_filter);
% note the transpose at the end of the line below!! reshape(..., [large_number, k * k]) is WRONG!
vectorized_1 = reshape(smoothed_gaussian_random_kernels_1, [k * k, large_number])';

% mean matrix (matricized)
mu_1 = mean(vectorized_1);

% cov matrix
sigma_smoothed_2 = cov(vectorized_1);

% ------------------------------------------------------------------------------

filter_width = 3;
gaussian_filter = fspecial('gaussian', [filter_width,filter_width], 1);
smoothed_gaussian_random_kernels_2 = imfilter(gaussian_random_kernels, gaussian_filter);
% note the transpose at the end of the line below!! reshape(..., [large_number, k * k]) is WRONG!
vectorized_2 = reshape(smoothed_gaussian_random_kernels_2, [k * k, large_number])';

% mean matrix (matricized)
mu_2 = mean(vectorized_2);

% cov matrix
sigma_smoothed_3 = cov(vectorized_2);

% ------------------------------------------------------------------------------

filter_width = 4;
gaussian_filter = fspecial('gaussian', [filter_width,filter_width], 1);
smoothed_gaussian_random_kernels_3 = imfilter(gaussian_random_kernels, gaussian_filter);
% note the transpose at the end of the line below!! reshape(..., [large_number, k * k]) is WRONG!
vectorized_3 = reshape(smoothed_gaussian_random_kernels_3, [k * k, large_number])';

% mean matrix (matricized)
mu_3 = mean(vectorized_3);

% cov matrix
sigma_smoothed_4 = cov(vectorized_3);

% ------------------------------------------------------------------------------

filter_width = 5;
gaussian_filter = fspecial('gaussian', [filter_width,filter_width], 1);
smoothed_gaussian_random_kernels_4 = imfilter(gaussian_random_kernels, gaussian_filter);
% note the transpose at the end of the line below!! reshape(..., [large_number, k * k]) is WRONG!
vectorized_4 = reshape(smoothed_gaussian_random_kernels_4, [k * k, large_number])';

% mean matrix (matricized)
mu_4 = mean(vectorized_4);

% cov matrix
sigma_smoothed_5 = cov(vectorized_4);

% ------------------------------------------------------------------------------

% for reference
sigma_identity = eye(25);

% setting mu
zero_filter = zeros(5,5);
mu = reshape(zero_filter, 1, 25);

close all
figure
colormap gray
for i = 1:2
  subplot(5, 6, 1 + 6 * (1 - 1)), imagesc(sigma_identity),                                           title('sigma idenity');
  subplot(5, 6, 2 + 6 * (1 - 1)), imagesc(reshape(mvnrnd(mu, sigma_identity, 1), 5, 5)),             title('sigma identity var / 1');
  subplot(5, 6, 3 + 6 * (1 - 1)), imagesc(reshape(mvnrnd(mu, sigma_identity / 10, 1), 5, 5)),        title('sigma identity var / 10');
  subplot(5, 6, 4 + 6 * (1 - 1)), imagesc(reshape(mvnrnd(mu, sigma_identity / 100, 1), 5, 5)),       title('sigma identity var / 100');
  subplot(5, 6, 5 + 6 * (1 - 1)), imagesc(reshape(mvnrnd(mu, sigma_identity / 1000, 1), 5, 5)),      title('sigma identity var / 1000');
  subplot(5, 6, 6 + 6 * (1 - 1)), imagesc(reshape(mvnrnd(mu, sigma_identity / 10000, 1), 5, 5)),     title('sigma identity var / 10000');

  subplot(5, 6, 1 + 6 * (2 - 1)), imagesc(sigma_smoothed_2),                                         title('sigma anisodiffed 2');
  subplot(5, 6, 2 + 6 * (2 - 1)), imagesc(reshape(mvnrnd(mu_1, sigma_smoothed_2, 1), 5, 5)),         title('sigma anisodiffed var / 1');
  subplot(5, 6, 3 + 6 * (2 - 1)), imagesc(reshape(mvnrnd(mu_1, sigma_smoothed_2 / 10, 1), 5, 5)),    title('sigma anisodiffed var / 10');
  subplot(5, 6, 4 + 6 * (2 - 1)), imagesc(reshape(mvnrnd(mu_1, sigma_smoothed_2 / 100, 1), 5, 5)),   title('sigma anisodiffed var / 100');
  subplot(5, 6, 5 + 6 * (2 - 1)), imagesc(reshape(mvnrnd(mu_1, sigma_smoothed_2 / 1000, 1), 5, 5)),  title('sigma anisodiffed var / 1000');
  subplot(5, 6, 6 + 6 * (2 - 1)), imagesc(reshape(mvnrnd(mu_1, sigma_smoothed_2 / 10000, 1), 5, 5)), title('sigma anisodiffed var / 10000');

  subplot(5, 6, 1 + 6 * (3 - 1)), imagesc(sigma_smoothed_3),                                         title('sigma anisodiffed 3');
  subplot(5, 6, 2 + 6 * (3 - 1)), imagesc(reshape(mvnrnd(mu_2, sigma_smoothed_3, 1), 5, 5)),         title('sigma anisodiffed var / 1');
  subplot(5, 6, 3 + 6 * (3 - 1)), imagesc(reshape(mvnrnd(mu_2, sigma_smoothed_3 / 10, 1), 5, 5)),    title('sigma anisodiffed var / 10');
  subplot(5, 6, 4 + 6 * (3 - 1)), imagesc(reshape(mvnrnd(mu_2, sigma_smoothed_3 / 100, 1), 5, 5)),   title('sigma anisodiffed var / 100');
  subplot(5, 6, 5 + 6 * (3 - 1)), imagesc(reshape(mvnrnd(mu_2, sigma_smoothed_3 / 1000, 1), 5, 5)),  title('sigma anisodiffed var / 1000');
  subplot(5, 6, 6 + 6 * (3 - 1)), imagesc(reshape(mvnrnd(mu_2, sigma_smoothed_3 / 10000, 1), 5, 5)), title('sigma anisodiffed var / 10000');

  subplot(5, 6, 1 + 6 * (4 - 1)), imagesc(sigma_smoothed_4),                                         title('sigma anisodiffed 4');
  subplot(5, 6, 2 + 6 * (4 - 1)), imagesc(reshape(mvnrnd(mu_3, sigma_smoothed_4, 1), 5, 5)),         title('sigma anisodiffed var / 1');
  subplot(5, 6, 3 + 6 * (4 - 1)), imagesc(reshape(mvnrnd(mu_3, sigma_smoothed_4 / 10, 1), 5, 5)),    title('sigma anisodiffed var / 10');
  subplot(5, 6, 4 + 6 * (4 - 1)), imagesc(reshape(mvnrnd(mu_3, sigma_smoothed_4 / 100, 1), 5, 5)),   title('sigma anisodiffed var / 100');
  subplot(5, 6, 5 + 6 * (4 - 1)), imagesc(reshape(mvnrnd(mu_3, sigma_smoothed_4 / 1000, 1), 5, 5)),  title('sigma anisodiffed var / 1000');
  subplot(5, 6, 6 + 6 * (4 - 1)), imagesc(reshape(mvnrnd(mu_3, sigma_smoothed_4 / 10000, 1), 5, 5)), title('sigma anisodiffed var / 10000');

  subplot(5, 6, 1 + 6 * (5 - 1)), imagesc(sigma_smoothed_5),                                         title('sigma anisodiffed 5');
  subplot(5, 6, 2 + 6 * (5 - 1)), imagesc(reshape(mvnrnd(mu_4, sigma_smoothed_5, 1), 5, 5)),         title('sigma anisodiffed var / 1');
  subplot(5, 6, 3 + 6 * (5 - 1)), imagesc(reshape(mvnrnd(mu_4, sigma_smoothed_5 / 10, 1), 5, 5)),    title('sigma anisodiffed var / 10');
  subplot(5, 6, 4 + 6 * (5 - 1)), imagesc(reshape(mvnrnd(mu_4, sigma_smoothed_5 / 100, 1), 5, 5)),   title('sigma anisodiffed var / 100');
  subplot(5, 6, 5 + 6 * (5 - 1)), imagesc(reshape(mvnrnd(mu_4, sigma_smoothed_5 / 1000, 1), 5, 5)),  title('sigma anisodiffed var / 1000');
  subplot(5, 6, 6 + 6 * (5 - 1)), imagesc(reshape(mvnrnd(mu_4, sigma_smoothed_5 / 10000, 1), 5, 5)), title('sigma anisodiffed var / 10000');
  pause(0.05);
end





















k = 5;
m = 1;
n = 1;
filter_width = 3;
init_multiplier = 1;

large_number = 100000;
gaussian_random_kernels = init_multiplier * randn(k, k, large_number, 'single');

% ------------------------------------------------------------------------------

d = 1000;
vectorized_1 = zeros(large_number, k * k);
for i = 1 : large_number
  g1 = gen2DGaussianFilter(5,1);
  g2 = g1 - mean(g1(:));
  g3 = g2 * sign(randn());
  g4 = g3 + randn(5,5) / d;
  vectorized_1(i,:) = reshape(g4, 1, k * k);
end

% mean matrix (matricized)
mu_1 = mean(vectorized_1);

% cov matrix
sigma_centre_surround_d_1000 = cov(vectorized_1);

% ------------------------------------------------------------------------------

d = 100;
vectorized_2 = zeros(large_number, k * k);
for i = 1 : large_number
  g1 = gen2DGaussianFilter(5,1);
  g2 = g1 - mean(g1(:));
  g3 = g2 * sign(randn());
  g4 = g3 + randn(5,5) / d;
  vectorized_2(i,:) = reshape(g4, 1, k * k);
end

% mean matrix (matricized)
mu_2 = mean(vectorized_2);

% cov matrix
sigma_centre_surround_d_100 = cov(vectorized_2);

% ------------------------------------------------------------------------------

d = 10;
vectorized_3 = zeros(large_number, k * k);
for i = 1 : large_number
  g1 = gen2DGaussianFilter(5,1);
  g2 = g1 - mean(g1(:));
  g3 = g2 * sign(randn());
  g4 = g3 + randn(5,5) / d;
  vectorized_3(i,:) = reshape(g4, 1, k * k);
end

% mean matrix (matricized)
mu_3 = mean(vectorized_3);

% cov matrix
sigma_centre_surround_d_10 = cov(vectorized_3);

% ------------------------------------------------------------------------------

d = 1;
vectorized_4 = zeros(large_number, k * k);
for i = 1 : large_number
  g1 = gen2DGaussianFilter(5,1);
  g2 = g1 - mean(g1(:));
  g3 = g2 * sign(randn());
  g4 = g3 + randn(5,5) / d;
  vectorized_4(i,:) = reshape(g4, 1, k * k);
end

% mean matrix (matricized)
mu_4 = mean(vectorized_4);

% cov matrix
sigma_centre_surround_d_1 = cov(vectorized_4);

% ------------------------------------------------------------------------------

% for reference
sigma_identity = eye(25);

% setting mu
zero_filter = zeros(5,5);
mu = reshape(zero_filter, 1, 25);

close all
figure
colormap gray
for i = 1:2
  subplot(5, 6, 1 + 6 * (1 - 1)), imagesc(sigma_identity),                                                       title('sigma idenity');
  subplot(5, 6, 2 + 6 * (1 - 1)), imagesc(reshape(mvnrnd(mu, sigma_identity, 1), 5, 5)),                         title('sigma identity var / 1');
  subplot(5, 6, 3 + 6 * (1 - 1)), imagesc(reshape(mvnrnd(mu, sigma_identity / 10, 1), 5, 5)),                    title('sigma identity var / 10');
  subplot(5, 6, 4 + 6 * (1 - 1)), imagesc(reshape(mvnrnd(mu, sigma_identity / 100, 1), 5, 5)),                   title('sigma identity var / 100');
  subplot(5, 6, 5 + 6 * (1 - 1)), imagesc(reshape(mvnrnd(mu, sigma_identity / 1000, 1), 5, 5)),                  title('sigma identity var / 1000');
  subplot(5, 6, 6 + 6 * (1 - 1)), imagesc(reshape(mvnrnd(mu, sigma_identity / 10000, 1), 5, 5)),                 title('sigma identity var / 10000');

  subplot(5, 6, 1 + 6 * (2 - 1)), imagesc(sigma_centre_surround_d_1000),                                         title('sigma centre surround - d 1000');
  subplot(5, 6, 2 + 6 * (2 - 1)), imagesc(reshape(mvnrnd(mu_1, sigma_centre_surround_d_1000, 1), 5, 5)),         title('sigma centre surround var / 1');
  subplot(5, 6, 3 + 6 * (2 - 1)), imagesc(reshape(mvnrnd(mu_1, sigma_centre_surround_d_1000 / 10, 1), 5, 5)),    title('sigma centre surround var / 10');
  subplot(5, 6, 4 + 6 * (2 - 1)), imagesc(reshape(mvnrnd(mu_1, sigma_centre_surround_d_1000 / 100, 1), 5, 5)),   title('sigma centre surround var / 100');
  subplot(5, 6, 5 + 6 * (2 - 1)), imagesc(reshape(mvnrnd(mu_1, sigma_centre_surround_d_1000 / 1000, 1), 5, 5)),  title('sigma centre surround var / 1000');
  subplot(5, 6, 6 + 6 * (2 - 1)), imagesc(reshape(mvnrnd(mu_1, sigma_centre_surround_d_1000 / 10000, 1), 5, 5)), title('sigma centre surround var / 10000');

  subplot(5, 6, 1 + 6 * (3 - 1)), imagesc(sigma_centre_surround_d_100),                                          title('sigma centre surround - d 100');
  subplot(5, 6, 2 + 6 * (3 - 1)), imagesc(reshape(mvnrnd(mu_2, sigma_centre_surround_d_100, 1), 5, 5)),          title('sigma centre surround var / 1');
  subplot(5, 6, 3 + 6 * (3 - 1)), imagesc(reshape(mvnrnd(mu_2, sigma_centre_surround_d_100 / 10, 1), 5, 5)),     title('sigma centre surround var / 10');
  subplot(5, 6, 4 + 6 * (3 - 1)), imagesc(reshape(mvnrnd(mu_2, sigma_centre_surround_d_100 / 100, 1), 5, 5)),    title('sigma centre surround var / 100');
  subplot(5, 6, 5 + 6 * (3 - 1)), imagesc(reshape(mvnrnd(mu_2, sigma_centre_surround_d_100 / 1000, 1), 5, 5)),   title('sigma centre surround var / 1000');
  subplot(5, 6, 6 + 6 * (3 - 1)), imagesc(reshape(mvnrnd(mu_2, sigma_centre_surround_d_100 / 10000, 1), 5, 5)),  title('sigma centre surround var / 10000');

  subplot(5, 6, 1 + 6 * (4 - 1)), imagesc(sigma_centre_surround_d_10),                                           title('sigma centre surround - d 10');
  subplot(5, 6, 2 + 6 * (4 - 1)), imagesc(reshape(mvnrnd(mu_3, sigma_centre_surround_d_10, 1), 5, 5)),           title('sigma centre surround var / 1');
  subplot(5, 6, 3 + 6 * (4 - 1)), imagesc(reshape(mvnrnd(mu_3, sigma_centre_surround_d_10 / 10, 1), 5, 5)),      title('sigma centre surround var / 10');
  subplot(5, 6, 4 + 6 * (4 - 1)), imagesc(reshape(mvnrnd(mu_3, sigma_centre_surround_d_10 / 100, 1), 5, 5)),     title('sigma centre surround var / 100');
  subplot(5, 6, 5 + 6 * (4 - 1)), imagesc(reshape(mvnrnd(mu_3, sigma_centre_surround_d_10 / 1000, 1), 5, 5)),    title('sigma centre surround var / 1000');
  subplot(5, 6, 6 + 6 * (4 - 1)), imagesc(reshape(mvnrnd(mu_3, sigma_centre_surround_d_10 / 10000, 1), 5, 5)),   title('sigma centre surround var / 10000');

  subplot(5, 6, 1 + 6 * (5 - 1)), imagesc(sigma_centre_surround_d_1),                                            title('sigma centre surround - d 1');
  subplot(5, 6, 2 + 6 * (5 - 1)), imagesc(reshape(mvnrnd(mu_4, sigma_centre_surround_d_1, 1), 5, 5)),            title('sigma centre surround var / 1');
  subplot(5, 6, 3 + 6 * (5 - 1)), imagesc(reshape(mvnrnd(mu_4, sigma_centre_surround_d_1 / 10, 1), 5, 5)),       title('sigma centre surround var / 10');
  subplot(5, 6, 4 + 6 * (5 - 1)), imagesc(reshape(mvnrnd(mu_4, sigma_centre_surround_d_1 / 100, 1), 5, 5)),      title('sigma centre surround var / 100');
  subplot(5, 6, 5 + 6 * (5 - 1)), imagesc(reshape(mvnrnd(mu_4, sigma_centre_surround_d_1 / 1000, 1), 5, 5)),     title('sigma centre surround var / 1000');
  subplot(5, 6, 6 + 6 * (5 - 1)), imagesc(reshape(mvnrnd(mu_4, sigma_centre_surround_d_1 / 10000, 1), 5, 5)),    title('sigma centre surround var / 10000');
  pause(0.05);
end





















































