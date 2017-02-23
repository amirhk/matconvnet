% without padding, stride 1
dim_image = 10;
dim_kernel = 3;
dim_output = dim_image - dim_kernel + 1;
I = ones(dim_image,dim_image);
K = 5 * ones(dim_kernel,dim_kernel);
disp('conv2 operation');
conv2(I,K, 'valid')

vec_I = reshape(I, [], 1);
toep = zeros(dim_output^2, dim_image^2);
for i = 1:dim_output^2
  toep(i, i:i+dim_output) = reshape(K, 1, []);
end
disp('toeplitz operation');
reshape(toep * vec_I, dim_output, dim_output)


% with padding, stride 1
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
