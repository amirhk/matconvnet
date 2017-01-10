% -------------------------------------------------------------------------
function compareWeights(input_opts)
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

networkArch = 'cifar-lenet';
layer = 4; % [1, 4, 7]
% networkArch = 'cifar-alexnet';
% layer = 1; % [1, 3, 6, 9, 11]

file_name = sprintf('W1-layer-%d.mat', layer);
devPath = getDevPath();

w_baseline = load(fullfile(devPath, 'data', networkArch, 'w_baseline', file_name));
w_compRand = load(fullfile(devPath, 'data', networkArch, 'w_compRand', file_name));
w_1D = load(fullfile(devPath, 'data', networkArch, 'w_1D', file_name));
w_2D_super = load(fullfile(devPath, 'data', networkArch, 'w_2D-super', file_name));
w_2D_posneg = load(fullfile(devPath, 'data', networkArch, 'w_2D-posneg', file_name));
w_2D_positive = load(fullfile(devPath, 'data', networkArch, 'w_2D-positive', file_name));
w_2D_shiftflip = load(fullfile(devPath, 'data', networkArch, 'w_2D-shiftflip', file_name));
w_2D_mult_randn = load(fullfile(devPath, 'data', networkArch, 'w_2D-mult-randn', file_name));
w_2D_mult_kernel = load(fullfile(devPath, 'data', networkArch, 'w_2D-mult-kernel', file_name));

w_baseline = w_baseline.W1;
w_compRand = w_compRand.W1;
w_1D = w_1D.W1;
w_2D_super = w_2D_super.W1;
w_2D_posneg = w_2D_posneg.W1;
w_2D_positive = w_2D_positive.W1;
w_2D_shiftflip = w_2D_shiftflip.W1;
w_2D_mult_randn = w_2D_mult_randn.W1;
w_2D_mult_kernel = w_2D_mult_kernel.W1;

assert(logical(prod(size(w_baseline) == size(w_compRand))));
assert(logical(prod(size(w_baseline) == size(w_1D))));
assert(logical(prod(size(w_baseline) == size(w_2D_super))));
assert(logical(prod(size(w_baseline) == size(w_2D_posneg))));
assert(logical(prod(size(w_baseline) == size(w_2D_positive))));
assert(logical(prod(size(w_baseline) == size(w_2D_shiftflip))));
assert(logical(prod(size(w_baseline) == size(w_2D_mult_randn))));
assert(logical(prod(size(w_baseline) == size(w_2D_mult_kernel))));

% randomly choose 3 kernels to compare
num_kernels = 9;
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
  w_2D_super_slice = w_2D_super(:, :, cc, dd);
  subplot(2,num_kernels,i), imshow(w_2D_super_slice, []), title('2D super');
  subplot(2,num_kernels,num_kernels + i), mesh(1:1:aa, 1:1:aa, w_2D_super_slice);
  i = i + 1;
  w_2D_posneg_slice = w_2D_posneg(:, :, cc, dd);
  subplot(2,num_kernels,i), imshow(w_2D_posneg_slice, []), title('2D posneg');
  subplot(2,num_kernels,num_kernels + i), mesh(1:1:aa, 1:1:aa, w_2D_posneg_slice);
  i = i + 1;
  w_2D_shiftflip_slice = w_2D_shiftflip(:, :, cc, dd);
  subplot(2,num_kernels,i), imshow(w_2D_shiftflip_slice, []), title('2D shiftflip');
  subplot(2,num_kernels,num_kernels + i), mesh(1:1:aa, 1:1:aa, w_2D_shiftflip_slice);
  i = i + 1;
  w_2D_mult_kernel_slice = w_2D_mult_kernel(:, :, cc, dd);
  subplot(2,num_kernels,i), imshow(w_2D_mult_kernel_slice, []), title('2D mult');
  subplot(2,num_kernels,num_kernels + i), mesh(1:1:aa, 1:1:aa, w_2D_mult_kernel_slice);
  i = i + 1;
  w_2D_mult_randn_slice = w_2D_mult_randn(:, :, cc, dd);
  subplot(2,num_kernels,i), imshow(w_2D_mult_randn_slice, []), title('2D mult 2');
  subplot(2,num_kernels,num_kernels + i), mesh(1:1:aa, 1:1:aa, w_2D_mult_randn_slice);
  % saveas(h, sprintf('Weight comparisons for slice (:,:,%d,%d).png', cc, dd));
end

