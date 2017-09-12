% -------------------------------------------------------------------------
function plotAndPrintSampleKernelsAndCovariance()
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

  k = 5;
  number_of_kernels = 10000;

  figure,

  % % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % % normal
  % % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % tmp_kernels = getTmpKernels(k, 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1', number_of_kernels);

  % % imagesc(reshape(tmp_kernels(1,:), k, k))
  % % set(findobj(gcf, 'type','axes'), 'Visible','off')
  % % export_fig 'sample_kernel_1_normal' -png

  % % imagesc(reshape(tmp_kernels(2,:), k, k))
  % % set(findobj(gcf, 'type','axes'), 'Visible','off')
  % % export_fig 'sample_kernel_2_normal' -png

  % % imagesc(reshape(tmp_kernels(3,:), k, k))
  % % set(findobj(gcf, 'type','axes'), 'Visible','off')
  % % export_fig 'sample_kernel_3_normal' -png

  % % imagesc(reshape(tmp_kernels(4,:), k, k))
  % % set(findobj(gcf, 'type','axes'), 'Visible','off')
  % % export_fig 'sample_kernel_4_normal' -png

  % imagesc(cov(tmp_kernels))
  % set(findobj(gcf, 'type','axes'), 'Visible','off')
  % export_fig 'covariance_normal' -png


  % % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % % log-normal
  % % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % tmp_kernels = getTmpKernels(k, 'logNormal-layer5-ratVisualCortex', number_of_kernels);

  % % imagesc(reshape(tmp_kernels(1,:), k, k))
  % % set(findobj(gcf, 'type','axes'), 'Visible','off')
  % % export_fig 'sample_kernel_1_log_normal' -png

  % % imagesc(reshape(tmp_kernels(2,:), k, k))
  % % set(findobj(gcf, 'type','axes'), 'Visible','off')
  % % export_fig 'sample_kernel_2_log_normal' -png

  % % imagesc(reshape(tmp_kernels(3,:), k, k))
  % % set(findobj(gcf, 'type','axes'), 'Visible','off')
  % % export_fig 'sample_kernel_3_log_normal' -png

  % % imagesc(reshape(tmp_kernels(4,:), k, k))
  % % set(findobj(gcf, 'type','axes'), 'Visible','off')
  % % export_fig 'sample_kernel_4_log_normal' -png

  % imagesc(cov(tmp_kernels))
  % set(findobj(gcf, 'type','axes'), 'Visible','off')
  % export_fig 'covariance_log_normal' -png


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % centre-surround
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  tmp_kernels = getTmpKernels(k, 'gaussian-CentreSurroundCovariance-randomDivide-100-MuDivide-1-SigmaDivide-1', number_of_kernels);

  imagesc(reshape(tmp_kernels(1,:), k, k))
  set(findobj(gcf, 'type','axes'), 'Visible','off')
  export_fig 'sample_kernel_1_centre_surround' -png

  imagesc(reshape(tmp_kernels(2,:), k, k))
  set(findobj(gcf, 'type','axes'), 'Visible','off')
  export_fig 'sample_kernel_2_centre_surround' -png

  imagesc(reshape(tmp_kernels(3,:), k, k))
  set(findobj(gcf, 'type','axes'), 'Visible','off')
  export_fig 'sample_kernel_3_centre_surround' -png

  imagesc(reshape(tmp_kernels(4,:), k, k))
  set(findobj(gcf, 'type','axes'), 'Visible','off')
  export_fig 'sample_kernel_4_centre_surround' -png

  imagesc(cov(tmp_kernels))
  set(findobj(gcf, 'type','axes'), 'Visible','off')
  export_fig 'covariance_centre_surround' -png


  % % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % % gaussian-blurred
  % % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % tmp_kernels = getTmpKernels(k, 'gaussian-SmoothedCovariance-3-MuDivide-1-SigmaDivide-1', number_of_kernels);

  % % imagesc(reshape(tmp_kernels(1,:), k, k))
  % % set(findobj(gcf, 'type','axes'), 'Visible','off')
  % % export_fig 'sample_kernel_1_gaussian_blurred' -png

  % % imagesc(reshape(tmp_kernels(2,:), k, k))
  % % set(findobj(gcf, 'type','axes'), 'Visible','off')
  % % export_fig 'sample_kernel_2_gaussian_blurred' -png

  % % imagesc(reshape(tmp_kernels(3,:), k, k))
  % % set(findobj(gcf, 'type','axes'), 'Visible','off')
  % % export_fig 'sample_kernel_3_gaussian_blurred' -png

  % % imagesc(reshape(tmp_kernels(4,:), k, k))
  % % set(findobj(gcf, 'type','axes'), 'Visible','off')
  % % export_fig 'sample_kernel_4_gaussian_blurred' -png

  % imagesc(cov(tmp_kernels))
  % set(findobj(gcf, 'type','axes'), 'Visible','off')
  % export_fig 'covariance_gaussian_blurred' -png








% -------------------------------------------------------------------------
function tmp_kernels = getTmpKernels(side_length, weight_init_type, number_of_kernels)
  % returns a matrix where each row is a vectorized kernel.
% -------------------------------------------------------------------------
  fh = networkInitializationUtils();
  structuredLayer = fh.convLayer('whatever', 'whatever', 1, side_length, number_of_kernels, 1, 1, 0, weight_init_type, 'gen');
  tmp_kernels = structuredLayer.weights{1};
  tmp_kernels = reshape(tmp_kernels, side_length^2, number_of_kernels)';
  % assert(size(tmp_kernel, 1) == side_length);
  % assert(size(tmp_kernel, 2) == side_length);







% side_length = 5;
% weight_init_type = 'gaussian-CentreSurroundCovariance-randomDivide-10-MuDivide-1-SigmaDivide-1';
% fh = networkInitializationUtils();
% structuredLayer = fh.convLayer('whatever', 'whatever', 1, side_length, 10000, 1, 1, 0, weight_init_type, 'gen');
% tmp_kernels = structuredLayer.weights{1};
% tmp_kernels = reshape(tmp_kernels, 25, 10000)';
% imagesc(cov(tmp_kernels))
% set(findobj(gcf, 'type','axes'), 'Visible','off')
% export_fig 'covariance_centre_surround' -png




















