% networkArch = 'cifar-lenet';
% layer = 4; % [1, 4, 7]
networkArch = 'cifar-alexnet';
layer = 11; % [1, 3, 6, 9, 11]

file_name = sprintf('W1-layer-%d.mat', layer);
devPath = getDevPath();

w_baseline = load(fullfile(devPath, 'data', networkArch, '+8epoch-baseline', file_name));
w_compRand = load(fullfile(devPath, 'data', networkArch, '+8epoch-compRand', file_name));
w_1D = load(fullfile(devPath, 'data', networkArch, '+8epoch-1D', file_name));
w_2D_mult = load(fullfile(devPath, 'data', networkArch, '+8epoch-2D-mult', file_name));
w_2D_super = load(fullfile(devPath, 'data', networkArch, '+8epoch-2D-super', file_name));
w_2D_posneg = load(fullfile(devPath, 'data', networkArch, '+8epoch-2D-posneg', file_name));
w_2D_positive = load(fullfile(devPath, 'data', networkArch, '+8epoch-2D-positive', file_name));
w_2D_amir = load(fullfile(devPath, 'data', networkArch, '+8epoch-2D-amir', file_name));

w_baseline = w_baseline.W1;
w_compRand = w_compRand.W1;
w_1D = w_1D.W1;
w_2D_mult = w_2D_mult.W1;
w_2D_super = w_2D_super.W1;
w_2D_posneg = w_2D_posneg.W1;
w_2D_positive = w_2D_positive.W1;
w_2D_amir = w_2D_amir.W1;

assert(logical(prod(size(w_baseline) == size(w_compRand))));
assert(logical(prod(size(w_baseline) == size(w_1D))));
assert(logical(prod(size(w_baseline) == size(w_2D_mult))));
assert(logical(prod(size(w_baseline) == size(w_2D_super))));
assert(logical(prod(size(w_baseline) == size(w_2D_posneg))));
assert(logical(prod(size(w_baseline) == size(w_2D_positive))));
assert(logical(prod(size(w_baseline) == size(w_2D_amir))));

% randomly choose 3 kernels to compare
num_kernels = 8;
for k = 1:2
  aa = size(w_baseline, 1);
  bb = size(w_baseline, 2);
  assert(aa == bb);
  cc = ceil(rand() * size(w_baseline, 3));
  dd = ceil(rand() * size(w_baseline, 4));
  fprintf('randomly choosing slice (:,:,%d,%d)\n', cc, dd);
  h = figure;
  i = 1;
  w_baseline_slice = w_baseline(:, :, cc, dd);
  subplot(2,num_kernels,i), imshow(w_baseline_slice, []), title('baseline');
  subplot(2,num_kernels,num_kernels + i), mesh(1:1:aa, 1:1:aa, w_baseline_slice);
  i = i + 1;
  w_compRand_slice = w_compRand(:, :, cc, dd);
  subplot(2,num_kernels,i), imshow(w_compRand_slice, []), title('random');
  subplot(2,num_kernels,num_kernels + i), mesh(1:1:aa, 1:1:aa, w_compRand_slice);
  i = i + 1;
  w_1D_slice = w_1D(:, :, cc, dd);
  subplot(2,num_kernels,i), imshow(w_1D_slice, []), title('1D');
  subplot(2,num_kernels,num_kernels + i), mesh(1:1:aa, 1:1:aa, w_1D_slice);
  i = i + 1;
  w_2D_positive_slice = w_2D_positive(:, :, cc, dd);
  subplot(2,num_kernels,i), imshow(w_2D_positive_slice, []), title('2D positive');
  subplot(2,num_kernels,num_kernels + i), mesh(1:1:aa, 1:1:aa, w_2D_positive_slice);
  i = i + 1;
  w_2D_mult_slice = w_2D_mult(:, :, cc, dd);
  subplot(2,num_kernels,i), imshow(w_2D_mult_slice, []), title('2D mult');
  subplot(2,num_kernels,num_kernels + i), mesh(1:1:aa, 1:1:aa, w_2D_mult_slice);
  i = i + 1;
  w_2D_super_slice = w_2D_super(:, :, cc, dd);
  subplot(2,num_kernels,i), imshow(w_2D_super_slice, []), title('2D super');
  subplot(2,num_kernels,num_kernels + i), mesh(1:1:aa, 1:1:aa, w_2D_super_slice);
  i = i + 1;
  w_2D_posneg_slice = w_2D_posneg(:, :, cc, dd);
  subplot(2,num_kernels,i), imshow(w_2D_posneg_slice, []), title('2D posneg');
  subplot(2,num_kernels,num_kernels + i), mesh(1:1:aa, 1:1:aa, w_2D_posneg_slice);
  i = i + 1;
  w_2D_amir_slice = w_2D_amir(:, :, cc, dd);
  subplot(2,num_kernels,i), imshow(w_2D_amir_slice, []), title('2D amir');
  subplot(2,num_kernels,num_kernels + i), mesh(1:1:aa, 1:1:aa, w_2D_amir_slice);
  % saveas(h, sprintf('Weight comparisons for slice (:,:,%d,%d).png', cc, dd));
end

