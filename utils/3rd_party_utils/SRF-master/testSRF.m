% Computer the nonlinear map based on SRF.
clear all;

%% kernel parameter
% a = 4;
% b = 7;
a = 2;
b = 1;
num_gaussians = 10;

%% data
load('usps.mat');
num_train = size(data_train, 1);
num_test = size(data_test,1);
X = [data_train; data_test];

%% normalization (each sample should have norm_2 = 1; this is NOT mean subtraction of each feature over all samples)
for i = 1:size(X,1)
    X(i,:) = X(i,:)./norm(X(i,:),2);
end

N_sample = 10000;
if (size(X,1) > N_sample)
    rand_r = randperm(size(X,1));
    X = X(rand_r(1:N_sample), :);
end


%% kernel approximation and evaluation
mapdim_all = 2.^[1:8];
% mapdim_all = 2.^[1:14];
kernelfunc = @(normz) (1 - (normz/a).^2 ).^b;
MSE = zeros(1, length(mapdim_all));
for ii = 1:length(mapdim_all)
    mapdim = mapdim_all(ii);
    fea = gen_RFF(X, a, b, num_gaussians, mapdim);
    kernel_gt = kernelfunc(sqrt(2 - X*X'*2));
    kernel_diff = (kernel_gt - fea * fea').^2;
    MSE(ii) = mean(kernel_diff(:));
end
close all;
figure;
hold on;
grid on;
plot(log(mapdim_all)/log(2), MSE);
xlabel('log(mapped dimensionality)');
ylabel('MSE')

