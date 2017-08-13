% without padding, stride 1
% my toeplitz implentation is identical to the conv2 operation!
dim_image = 10;
dim_kernel = 3;
dim_output = dim_image - dim_kernel + 1;
I = ones(dim_image,dim_image);
K = 5 * ones(dim_kernel,dim_kernel);
disp('conv2 operation');
conv2(I, K, 'valid')

vec_I = reshape(I, [], 1);
toep = zeros(dim_output^2, dim_image^2);
for i = 1:dim_output^2
  toep(i, i:i+dim_output) = reshape(K, 1, []);
end
disp('toeplitz operation');
reshape(toep * vec_I, dim_output, dim_output)


% with padding, stride 1
% my toeplitz implentation IS NOT identical to the conv2 operation
dim_image = 10;
dim_kernel = 3;
dim_output = dim_image - dim_kernel + 1;
I = ones(dim_image,dim_image);
K = 5 * ones(dim_kernel,dim_kernel);
disp('conv2 operation');
conv2(I,K, 'same')

vec_I = reshape(I, [], 1);
toep = zeros(dim_image^2, dim_image^2);
for i = 1:dim_image^2
  row_indices = i:i+dim_output;                  % e.g. [98    99   100   101   102   103   104   105   106]
  row_indices = mod(row_indices, dim_image^2);   % e.g. [98    99     0     1     2     3     4     5     6]
  row_indices(find(~row_indices)) = dim_image^2; % e.g. [98    99   100     1     2     3     4     5     6]
  toep(i, row_indices) = reshape(K, 1, []);
end
disp('toeplitz operation');
reshape(toep * vec_I, dim_image, dim_image)

























% with padding, stride 1
% my toeplitz implentation IS NOT identical to the conv2 operation
stride = 2;
dim_image = 10;
dim_kernel = 5;
assert(mod(dim_kernel, 2) == 1, 'expect dim_kernel to be an odd number.');
dim_padding = (dim_kernel - 1) / 2;
dim_output = dim_image / stride;
% I = ones(dim_image, dim_image);
I = round(rand(dim_image) * 100);
K = 5 * ones(dim_kernel, dim_kernel);
disp('conv2 operation');
conv2(I,K, 'same');


padded_image = padarray(I, [dim_padding, dim_padding], 'symmetric', 'both');
assert(size(padded_image, 1) == dim_image + 2 * dim_padding, 'padded image has messed up dimensions.');
vectorized_padded_image = reshape(padded_image, [], 1);
toeplitz_matrix_dim_y = dim_output ^ 2;
toeplitz_matrix_dim_x = (dim_image + 2 * dim_padding) ^ 2;
toeplitx_matrix_mask = zeros(toeplitz_matrix_dim_y, toeplitz_matrix_dim_x);


% [1 1 1 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 1 1 1 0 0 0 ... 0 0 0]
toeplitz_row_mask = zeros(1, toeplitz_matrix_dim_x);
tmp_index = 1;
for i = 1 : dim_kernel
  toeplitz_row_mask(tmp_index : tmp_index + dim_kernel - 1) = 1;
  toeplitz_row_mask(tmp_index + dim_kernel : tmp_index + dim_image + 2 * dim_padding - 1) = 0;
  tmp_index = tmp_index + dim_image + 2 * dim_padding;
end

for i = 1 : toeplitz_matrix_dim_y
  toeplitx_matrix_mask(i,:) = circshift(toeplitz_row_mask, i - 1);
end


figure,

subplot(1,2,1),
imshow(toeplitx_matrix_mask)
ylabel(size(toeplitx_matrix_mask,1))
xlabel(size(toeplitx_matrix_mask,2))

subplot(1,2,2),
imshow(vectorized_padded_image, [])
ylabel(size(vectorized_padded_image,1))
xlabel(size(vectorized_padded_image,2))


% disp('toeplitz operation');
% reshape(toep * vec_I, dim_image, dim_image)






































































