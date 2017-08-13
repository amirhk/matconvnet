% -------------------------------------------------------------------------
function toeplitz_matrix_mask = createToeplitzMask(dim_image, dim_kernel, debug_flag)
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


  % padding: [TOP BOTTOM LEFT RIGHT]
  padding = [(dim_kernel - 1) / 2, (dim_kernel - 1) / 2, (dim_kernel - 1) / 2, (dim_kernel - 1) / 2];
  stride = 1;

  assert(mod(dim_kernel, 2) == 1, 'expect dim_kernel to be an odd number.');
  dim_output = dim_image / stride;
  I = round(rand(dim_image));
  % I = ones(dim_image);

  padded_image = padImageUsingPadding(I, padding);
  assert(size(padded_image, 1) == dim_image + padding(1) + padding(2), 'padded image has messed up dimensions.');
  assert(size(padded_image, 2) == dim_image + padding(3) + padding(4), 'padded image has messed up dimensions.');

  vectorized_padded_image = reshape(padded_image, [], 1);
  toeplitz_matrix_dim_y = dim_output ^ 2;
  toeplitz_matrix_dim_x = size(vectorized_padded_image, 1);
  toeplitz_matrix_mask = zeros(toeplitz_matrix_dim_y, toeplitz_matrix_dim_x);

  % [1 1 1 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 1 1 1 0 0 0 ... 0 0 0]
  toeplitz_row_mask = zeros(1, toeplitz_matrix_dim_x);
  tmp_index = 1;
  for i = 1 : dim_kernel
    toeplitz_row_mask(tmp_index : tmp_index + dim_kernel - 1) = 1;
    toeplitz_row_mask(tmp_index + dim_kernel : tmp_index + dim_image + padding(3) + padding(4) - 1) = 0;
    tmp_index = stride *  (tmp_index + dim_image + padding(3) + padding(4));
    % toeplitz_row_mask(tmp_index + dim_kernel : tmp_index + dim_image + 2 * dim_padding - 1) = 0;
    % tmp_index = tmp_index + dim_image + 2 * dim_padding;
  end

  tmp_image_coeff_counter = 1;
  tmp_offset = 0;
  for i = 1 : toeplitz_matrix_dim_y
    toeplitz_matrix_mask(i,:) = circshift(toeplitz_row_mask, i - 1 + tmp_offset);
    % if we have convolved with all image coefficients in this row,....
    if mod(tmp_image_coeff_counter, dim_image) == 0
      % jump 1 row below
      % tmp_offset = tmp_offset + (dim_image + padding(3) + padding(4));
      tmp_offset = tmp_offset + dim_kernel - 1;
    end
    tmp_image_coeff_counter = tmp_image_coeff_counter + 1;
  end

  if debug_flag
    K = 5 * ones(dim_kernel, dim_kernel);
    toeplitx_matrix = 5 * ones(size(toeplitz_matrix_mask)) .* toeplitz_matrix_mask;
    figure,

    subplot(3,2,1),
    imshow(I, []);
    ylabel(size(I,1)),
    xlabel(size(I,2));
    title('image')

    subplot(3,2,2),
    imshow(K, []);
    ylabel(size(K,1)),
    xlabel(size(K,2));
    title('kernel');

    subplot(3,2,3),
    imshow(toeplitz_matrix_mask),
    ylabel(size(toeplitz_matrix_mask,1)),
    xlabel(size(toeplitz_matrix_mask,2));
    title('toeplitx matrix mask');

    subplot(3,2,4),
    imshow(vectorized_padded_image, []),
    ylabel(size(vectorized_padded_image,1)),
    xlabel(size(vectorized_padded_image,2));
    title('vectorized padded image');

    subplot(3,2,5),
    tmp = conv2(I, K, 'same');
    imshow(tmp, []);
    ylabel(size(tmp,1)),
    xlabel(size(tmp,2));
    title('conv2 operation');

    subplot(3,2,6),
    tmp = reshape(toeplitx_matrix * vectorized_padded_image, dim_output, dim_output);
    imshow(tmp, []);
    ylabel(size(tmp,1)),
    xlabel(size(tmp,2));
    title('toeplitz operation');
  end




















