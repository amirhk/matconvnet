function learning_rate = getLearningRate(dataset, network_arch)
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

switch network_arch
  % case 'lenet_bu'
  %   switch dataset
  %     % multi-class
  %     case 'cifar'
  %       learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
  %     case 'coil-100'
  %       learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
  %     case 'mnist'
  %       learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
  %     case 'stl-10'
  %       learning_rate = [0.5*ones(1,20) 0.05*ones(1,15)  0.1:-0.01:0.06 0.05*ones(1,10)];
  %     case 'svhn'
  %       learning_rate = [0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,10)];
  %     case 'cifar-two-class-deer-horse'
  %       learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,120)];
  %     case 'cifar-two-class-deer-truck'
  %       learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,120)];
  %     case 'cifar-no-white-two-class-deer-truck'
  %       learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,120)];

  %     % multi-class subsampled
  %     case 'mnist-multi-class-subsampled'
  %       learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
  %     case 'svhn-multi-class-subsampled'
  %       learning_rate = [0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,10)];
  %     case 'cifar-multi-class-subsampled'
  %       learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,25)];
  %     case 'stl-10-multi-class-subsampled'
  %       learning_rate = [0.5*ones(1,20) 0.05*ones(1,15)  0.1:-0.01:0.06 0.05*ones(1,10)];

  %     % two-class
  %     case 'mnist-two-class-9-4'
  %       learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,120)];
  %     case 'stl-10-two-class-airplane-bird'
  %       learning_rate = [0.05*ones(1,20) 0.005*ones(1,20) 0.001*ones(1,110)] * 10;
  %     case 'stl-10-two-class-airplane-cat'
  %       learning_rate = [0.05*ones(1,20) 0.005*ones(1,20) 0.001*ones(1,110)] * 10;
  %       % learning_rate = [0.05*ones(1,20) 0.005*ones(1,20) 0.001*ones(1,100) 0.0001*ones(1,310)] * 10;
  %     case 'svhn-two-class-9-4'
  %       learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,120)];
  %     case 'prostate-v2-20-patients'
  %       learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20)];
  %     case 'prostate-v3-104-patients'
  %       learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20)];
  %   end


  case 'larpV0P0+convV0P0+fcV1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 09.87 / 09.08 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 99.14 / 95.49 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 95.83 / 94.29 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 83.83 / 77.09 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 83.08 / 77.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 76.94 / 73.69 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 01.00  / 01.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 01.00  / 01.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 100.00 / 99.75 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 100.00 / 99.19 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 100.00 / 96.83 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;    % 99.74 / 42.96 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;    % 95.26 / 43.70 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 69.34 / 44.08 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 48.26 / 41.20 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 38.26 / 36.29 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 30.38 / 30.13 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 24.12 / 23.89 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 10.00 / 10.00 (non-smooth random kernels; batch size:  50; weight decay: 0.01)
                                                                                       % 89.98 / 40.46 (non-smooth random kernels; batch size: 100; weight decay: 0.01) % BEST
                                                                                       % 87.39 / 37.74 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 91.31 / 38.80 (non-smooth random kernels; batch size:  50; weight decay: 0.01)
                                                                                       % 73.49 / 36.46 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
                                                                                       % 60.91 / 36.43 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 24.08 / 22.48 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)

        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 10.00 / 10.00 (smoothed random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 10.00 / 10.00 (smoothed random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 91.24 / 38.83 (smoothed random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 82.26 / 38.52 (smoothed random kernels; batch size: 100; weight decay: 0.0001)
    end


  case 'larpV1P0+convV0P0+fcV1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 09.87 / 09.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 99.29 / 96.96 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 99.75 / 96.91 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 98.08 / 96.48 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 94.92 / 94.06 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 69.94 / 62.57 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 85.41 / 74.64 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 84.63 / 75.38 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 79.83 / 73.92 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 70.95 / 67.25 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 04.00  / 04.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 100.00 / 99.75 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 100.00 / 99.19 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 99.53  / 97.69 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 90.83  / 89.75 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;    % 99.06 / 50.43 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;    % 72.84 / 50.13 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 48.64 / 43.51 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 35.46 / 35.05 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 26.54 / 26.29 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 18.90 / 19.18 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 12.22 / 12.28 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 37.30 / 29.05 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 79.02 / 48.28 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 67.75 / 45.66 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 44.71 / 39.64 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 29.28 / 27.79 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)

        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 98.22 / 57.70 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 99.97 / 57.36 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 95.14 / 57.29 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 44.57 / 39.63 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
    end
  case 'larpV1P1+convV0P0+fcV1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 09.87 / 09.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 95.48 / 94.73 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 99.34 / 97.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 98.29 / 97.26 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 96.40 / 95.94 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 81.19 / 74.16 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 88.21 / 78.70 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 86.33 / 78.54 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 82.20 / 77.29 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 75.59 / 72.12 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 32.19  / 31.92 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 100.00 / 99.92 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 100.00 / 99.75 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 99.61  / 98.94 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 90.42  / 89.81 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;    % 88.60 / 56.99 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;    % 60.00 / 52.66 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 44.78 / 42.95 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 31.68 / 31.81 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 25.36 / 24.66 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 18.18 / 18.08 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 12.72 / 13.05 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 61.46 / 51.37 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 66.64 / 57.18 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 57.13 / 52.99 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 44.28 / 42.87 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)

        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 77.99 / 60.20 (smoothed random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 98.15 / 63.74 (smoothed random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 89.58 / 64.14 (smoothed random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 73.03 / 63.00 (smoothed random kernels; batch size: 100; weight decay: 0.0001)
    end


  case 'larpV3P0+convV0P0+fcV1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 20.99 / 21.08 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 99.45 / 96.93 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 99.52 / 96.57 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 97.48 / 96.01 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 93.82 / 93.46 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 57.77 / 52.74 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 86.64 / 73.43 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 83.06 / 73.67 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 76.26 / 70.95 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 66.15 / 62.87 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 03.00 / 03.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 89.86 / 88.22 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 99.69 / 98.58 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 98.03 / 95.83 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 84.28 / 82.75 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;    % 98.66 / 48.68 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;    % 66.64 / 48.11 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 43.76 / 40.75 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 33.80 / 33.26 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 22.74 / 22.66 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 16.78 / 16.41 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 13.16 / 12.60 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 69.61 / 39.71 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 56.87 / 41.42 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 37.90 / 34.28 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 27.99 / 27.56 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)

        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 88.63 / 60.48 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 88.21 / 60.22 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 72.26 / 59.29 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 37.76 / 34.20 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
    end
  case 'larpV3P1+convV0P0+fcV1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 21.01 / 21.06 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 99.61 / 98.23 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 99.52 / 98.08 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 98.31 / 97.28 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 95.96 / 95.71 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 85.75 / 74.86 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 88.66 / 76.79 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 84.04 / 76.10 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 77.82 / 73.11 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 69.26 / 66.03 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 91.08  / 90.47 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 100.00 / 99.94 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 99.97  / 99.81 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 98.36  / 98.42 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 85.44  / 85.19 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;    % 65.48 / 49.62 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;    % 44.80 / 42.06 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 33.70 / 33.73 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 24.50 / 24.69 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 17.98 / 18.31 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 16.90 / 16.46 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 12.64 / 12.09 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 50.50 / 46.50 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 78.63 / 58.58 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 66.27 / 57.26 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 55.29 / 51.97 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 44.25 / 43.13 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)

        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 77.91 / 63.19 (smoothed random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 74.55 / 64.04 (smoothed random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 65.85 / 60.69 (smoothed random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 57.48 / 55.29 (smoothed random kernels; batch size: 100; weight decay: 0.0001)
    end
  case 'larpV3P3+convV0P0+fcV1' % = lenet_bu
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 09.87 / 09.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 97.92 / 97.44 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 97.67 / 97.47 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 95.85 / 95.70 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 91.87 / 92.50 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 75.16 / 70.83 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 75.32 / 71.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 71.10 / 68.43 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 63.95 / 63.03 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 50.66 / 50.94 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 99.97 / 100.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 99.69 / 99.72 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 96.00 / 95.94 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 85.19 / 85.92 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 34.72 / 34.97 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;    % 37.34 / 38.05 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;    % 26.68 / 26.85 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 17.02 / 17.06 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 15.30 / 14.68 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 14.80 / 14.85 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 11.26 / 11.29 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 11.16 / 11.14 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.05*ones(1,10) 0.05:-0.01:0.01 0.01*ones(1,5)  0.005*ones(1,10) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,4)]; % (bpd 13) javad LR w/ weight decay 0.01: 90.23 / 83.02
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 48.17 / 47.63 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 58.42 / 57.58 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 57.27 / 56.37 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 52.91 / 52.86 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 45.91 / 46.11 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)

        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 60.73 / 58.73 (smoothed random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 58.68 / 57.23 (smoothed random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 52.80 / 52.22 (smoothed random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 44.37 / 43.94 (smoothed random kernels; batch size: 100; weight decay: 0.0001)


      % case 'stl-10'
      %   % learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];                                        % 15.05 / 14.69
      %   % learning_rate = [0.5*ones(1,20) 0.05*ones(1,15)  0.1:-0.01:0.06 0.05*ones(1,10)];                            % 27.15 / 27.48
      %   % learning_rate = [0.5*ones(1,50) 0.05*ones(1,100)];                                                           % 36.10 / 36.60
      %   % learning_rate = [0.5*ones(1,150)];                                                                           % 45.32 / 44.85
      %   learning_rate = [1 * ones(1,20) 0.5*ones(1,130)];                                                              % 46.48 / 45.55
    end


  case 'larpV5hP0+convV0P0+fcV1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 09.87 / 09.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 20.94 / 21.04 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 92.47 / 91.24 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 91.16 / 90.94 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 18.92 / 19.60 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 32.30 / 32.68 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 58.14 / 52.84 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 55.48 / 52.22 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 01.00 / 01.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 01.00 / 01.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 01.00 / 01.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 01.75 / 02.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 03.28 / 03.78 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;    % 31.06 / 22.64 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;    % 93.58 / 46.88 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 68.34 / 46.02 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 46.10 / 40.41 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 34.30 / 33.63 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 24.28 / 24.29 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 17.14 / 17.14 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 23.54 / 23.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 24.75 / 23.84 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 24.00 / 23.46 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
    end
  case 'larpV5hP1+convV0P0+fcV1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 09.87 / 09.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 09.87 / 09.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 20.98 / 21.05 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 97.61 / 96.28 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 95.61 / 95.50 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 18.93 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 72.87 / 66.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 71.54 / 66.19 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 63.56 / 60.31 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 01.00 / 01.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 01.00 / 01.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 88.25 / 88.94 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 81.28 / 81.53 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 24.11 / 25.06 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;    % 61.56 / 46.10 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;    % 61.94 / 48.68 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 47.10 / 43.06 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 36.66 / 35.38 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 24.54 / 24.09 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 13.94 / 14.20 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 11.88 / 11.54 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 24.56 / 25.05 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 54.84 / 51.61 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 44.37 / 43.39 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 37.34 / 37.11 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
    end
  case 'larpV5hP3+convV0P0+fcV1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 09.87 / 09.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 20.85 / 20.98 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 95.69 / 95.72 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 94.47 / 94.90 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 68.53 / 65.04 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 68.20 / 66.11 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 59.22 / 58.56 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 01.00 / 01.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 80.86 / 81.36 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 96.14 / 96.39 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 89.36 / 90.14 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 66.97 / 68.33 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;    % 45.70 / 44.10 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;    % 37.00 / 37.11 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 27.76 / 28.14 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 16.42 / 15.68 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 11.34 / 10.64 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 10.83 / 10.94 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 47.67 / 47.03 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 51.75 / 51.35 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 49.61 / 48.91 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
    end
  case 'larpV5hP5+convV0P0+fcV1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 84.92 / 85.32 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 84.82 / 84.95 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 81.72 / 82.17 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 75.67 / 76.83 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 61.69 / 62.68 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 37.39 / 35.55 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 34.30 / 33.98 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 26.94 / 28.63 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 23.42 / 24.74 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 18.94 / 19.63 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 83.97 / 85.17 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 61.14 / 61.47 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 11.22 / 10.81 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 04.58 / 04.67 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 01.19 / 01.36 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;    % 10.64 / 10.95 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;    % 13.58 / 13.58 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 11.82 / 11.76 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 12.38 / 12.51 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 38.51 / 38.75 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 39.41 / 39.83 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 36.46 / 36.85 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 31.00 / 30.36 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 20.12 / 19.77 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
    end


  case 'larpV5aP0+convV0P0+fcV1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 91.40 / 91.64 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 82.74 / 83.66 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 55.05 / 55.21 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 22.44 / 22.52 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 19.75 / 20.43 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 36.03 / 36.66 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 04.06 / 03.92 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 02.25 / 02.08 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 01.47 / 01.39 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 01.03 / 00.83 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 00.97 / 00.89 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;    % << didn't test >> (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;    % 08.68 / 08.44 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 08.38 / 08.43 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 08.36 / 08.24 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 08.30 / 08.26 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 08.08 / 07.78 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 08.14 / 07.93 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 26.85 / 26.62 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 16.87 / 16.63 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 11.34 / 11.35 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 10.53 / 10.13 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
    end
  case 'larpV5aP1+convV0P0+fcV1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 89.95 / 90.71 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 79.88 / 80.79 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 40.64 / 40.85 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 26.55 / 27.02 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 25.55 / 25.49 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 18.79 / 19.44 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 01.89 / 01.86 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 01.58 / 01.08 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 01.47 / 00.11 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 01.06 / 00.69 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 00.67 / 01.08 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;    % << didn't test >> (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;    % 11.18 / 11.58 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 10.94 / 11.24 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 10.48 / 10.63 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 10.90 / 11.14 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 10.76 / 10.69 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 10.58 / 10.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 26.37 / 26.40 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 16.01 / 15.88 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
    end
  case 'larpV5aP3+convV0P0+fcV1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 76.44 / 77.97 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 31.51 / 31.18 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 11.06 / 11.11 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 14.79 / 15.46 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 09.85 / 10.08 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 18.92 / 19.58 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 13.13 / 13.73 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 01.00 / 01.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 01.03 / 01.06 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 00.92 / 00.97 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 00.97 / 00.89 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 01.00 / 00.89 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;    % << didn't test >> (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;    % 09.20 / 09.16 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 09.36 / 09.13 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 09.30 / 09.14 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 09.46 / 09.13 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 09.36 / 09.23 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 09.32 / 09.23 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 15.49 / 15.32 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 10.18 / 10.13 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
    end
  case 'larpV5aP5+convV0P0+fcV1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 16.14 / 16.41 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 14.62 / 14.30 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 10.31 / 10.07 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 12.15 / 12.34 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 10.10 / 10.13 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 14.69 / 16.24 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 13.98 / 15.23 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 01.00 / 01.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 01.00 / 01.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 00.92 / 00.94 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 00.94 / 00.97 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 00.86 / 00.94 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;    % << didn't test >> (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;    % 11.20 / 11.06 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 10.94 / 11.01 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 11.22 / 11.08 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 11.38 / 10.96 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 11.26 / 10.91 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 11.08 / 11.14 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;    % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;    % 11.18 / 11.28 (non-smooth random kernels; batch size: 100; weight decay: 0.0001) % BEST
    end


%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%


  case 'larpV0P0+convV0P0+fcV2'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 09.87 / 09.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 09.87 / 09.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 98.96 / 96.26 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 99.29 / 94.79 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 83.89 / 73.92 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 90.05 / 75.76 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 85.07 / 73.65 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 01.00 / 01.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 02.44 / 02.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 96.78 / 95.31 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 99.53 / 96.56 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
                                                                                           % 99.03 / 96.22 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;        % 99.06 / 93.36 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 300;        % 95.83 / 88.28 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;        % 99.86 / 45.48 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;        % 100.00 / 46.21 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 99.20 / 45.28 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 62.96 / 44.24 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 41.66 / 38.10 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 32.82 / 31.81 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 10.67 / 10.25 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 93.64 / 35.61 (non-smooth random kernels; batch size:  50; weight decay: 0.01)   % GOLD
                                                                                           % 25.08 / 17.01 (non-smooth random kernels; batch size:  50; weight decay: 0.0001) % BRONZE
                                                                                           % 45.23 / 26.11 (non-smooth random kernels; batch size: 100; weight decay: 0.01)   % SILVER
                                                                                           % 20.80 / 15.68 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
                                                                                           % 20.06 / 16.45 (non-smooth random kernels; batch size: 250; weight decay: 0.01)
                                                                                           % 15.89 / 13.84 (non-smooth random kernels; batch size: 250; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;        % 19.99 / 17.77 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
                                                                                           % 20.54 / 17.93 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 300;        % 14.48 / 12.98 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
                                                                                           % 14.63 / 13.18 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)

        % 75+ epochs:
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,225)] / 030;        % 80.74 / 29.49 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
                                                                                            % 32.49 / 18.70 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [                 0.03*ones(1,25) 0.01*ones(1,250)] / 030;        % 59.41 / 29.06 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
                                                                                            % 39.91 / 24.69 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
    end


  case 'larpV1P0+convV0P0+fcV2'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 09.87 / 09.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 99.96 / 97.43 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 100.00 / 96.99 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 99.78 / 96.16 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 99.26 / 77.32 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 94.72 / 76.04 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 84.67 / 74.85 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 01.00 / 01.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 100.00 / 99.94 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 100.00 / 99.42 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 99.97 / 98.31 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;        % 100.00 / 50.94 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;        % 99.82 / 50.48 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 65.60 / 49.01 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 40.64 / 38.70 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 28.92 / 28.08 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 20.48 / 19.85 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 98.48 / 47.58 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 84.72 / 45.75 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 53.58 / 42.64 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
    end
  case 'larpV1P1+convV0P0+fcV2'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 09.87 / 09.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 09.87 / 09.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 99.98 / 97.90 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 99.66 / 97.62 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 18.92 / 19.58 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 98.63 / 80.79 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 93.23 / 79.43 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 85.81 / 78.28 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 01.00 / 01.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 100.00 / 100.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 100.00 / 99.97 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 99.97 / 99.50 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;        % 100.00 / 57.29 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;        % 92.00 / 57.29 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 56.12 / 50.61 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 39.30 / 38.88 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 26.78 / 26.40 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 24.36 / 24.30 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 82.05 / 59.46 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 64.44 / 56.47 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
    end


  case 'larpV3P0+convV0P0+fcV2'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 09.87 / 09.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 09.87 / 09.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 99.95 / 96.82 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 99.22 / 95.90 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 18.92 / 19.59 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 98.99 / 76.31 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 91.53 / 73.64 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 80.69 / 72.45 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 01.00 / 01.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 100.00 / 99.44 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 100.00 / 99.33 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 99.44 / 98.06 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;        % 100.00 / 48.79 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;        % 98.98 / 48.15 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 56.82 / 44.73 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 36.88 / 35.68 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 25.80 / 25.36 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 18.16 / 18.38 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 99.48 / 45.83 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 71.02 / 41.97 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 43.35 / 36.88 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
    end
  case 'larpV3P1+convV0P0+fcV2'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 09.87 / 09.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 100.00 / 98.55 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 99.96 / 98.01 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 99.28 / 97.51 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 98.82 / 79.68 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 98.15 / 77.66 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 90.26 / 76.42 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 81.50 / 74.17 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 01.00 / 01.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 100.00 / 99.97 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 100.00 / 99.89 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 99.86 / 99.53 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;        % 98.52 / 50.70 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;        % 61.74 / 48.68 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 39.00 / 37.98 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 28.42 / 28.78 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 17.66 / 17.20 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 18.26 / 17.79 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 98.84 / 62.21 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 97.87 / 59.52 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 77.09 / 57.62 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 60.42 / 54.25 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
    end
  case 'larpV3P3+convV0P0+fcV2' % = lenet_bu but w/ larger FC
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 98.85 / 98.30 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 99.12 / 98.44 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 98.50 / 98.15 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 97.26 / 96.95 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 83.51 / 76.61 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 81.57 / 75.66 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 76.36 / 72.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 69.24 / 67.02 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 100.00 / 100.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 99.94 / 99.97 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 98.61 / 98.69 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 91.25 / 91.03 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;        % 49.58 / 46.84 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;        % 34.28 / 33.25 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 23.74 / 24.51 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 16.14 / 15.64 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 12.90 / 12.23 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 13.04 / 12.61 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;        % 65.04 / 59.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;        % 66.38 / 61.37 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;        % 62.25 / 59.30 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;        % 56.88 / 55.85 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
    end


%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%


  case 'larpV0sP0+convV1sP1+fcV1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;            % 09.87 / 09.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;            % 09.87 / 09.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;            % 09.87 / 09.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;            % 99.35 / 98.18 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;            % 10.00 / 10.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;            % 10.01 / 09.99 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;            % 74.89 / 65.28 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;            % 70.81 / 64.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
    end
  case 'larpV1sP0+convV1sP1+fcV1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;            % 100.00 / 98.99 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;            % 100.00 / 98.84 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;            % 99.80 / 98.52 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;            % 98.93 / 98.02 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;            % 99.90 / 62.77 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;            % 98.79 / 61.52 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;            % 75.51 / 63.38 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;            % 65.97 / 60.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
    end
  case 'larpV2sP0+convV1sP1+fcV1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;            % 99.89 / 98.41 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;            % 98.93 / 97.73 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;            % 96.43 / 96.02 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;            % 92.23 / 92.68 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;            % 94.35 / 56.48 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;            % 66.66 / 57.77 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;            % 50.72 / 49.05 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;            % 28.75 / 28.76 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
    end
  case 'larpV1lP0+convV1lP1+fcV1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;            % 09.87 / 09.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;            % 99.99 / 98.69 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;            % 100.00 / 98.73 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;            % 99.93 / 98.47 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;            % 88.67 / 61.07 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;            % 99.38 / 66.96 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;            % 94.05 / 68.46 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;            % 86.08 / 66.47 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
    end
  case 'larpV2lP0+convV1lP1+fcV1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;            % 100.00 / 98.95 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;            % 100.00 / 98.95 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;            % 99.99 / 98.57 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;            % 99.66 / 98.33 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;            % 99.73 / 67.12 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;            % 99.82 / 68.68 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;            % 93.91 / 65.50 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;            % 80.14 / 64.87 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
    end


%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%


  case 'larpV0P0+convV3lP1+fcV1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;            % 09.87 / 09.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;            % 09.87 / 09.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;            % 99.99 / 98.87 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;            % 99.95 / 98.71 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;            % 80.12 / 55.00 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;            % 99.64 / 72.21 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;            % 95.04 / 72.67 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;            % 82.54 / 73.17 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
    end
  case 'larpV1lP0+convV3lP1+fcV1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;            % 09.87 / 09.80 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;            % 100.00 / 99.08 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;            % 100.00 / 98.96 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;            % 99.91 / 98.87 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;            % 99.77 / 69.72 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;            % 100.00 / 75.64 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;            % 94.21 / 74.78 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;            % 84.49 / 73.99 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
    end
  case 'larpV2lP0+convV3lP1+fcV1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;            % 100.00 / 99.17 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;            % 100.00 / 99.07 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;            % 99.98 / 98.96 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;            % 99.69 / 98.69 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;            % 100.00 / 74.08 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;            % 99.92 / 74.90 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;            % 91.06 / 75.16 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;            % 79.17 / 71.58 (non-smooth random kernels; batch size: 100; weight decay: 0.0001)
    end


%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%


  case 'larpV0P0+convV3P3+fcV1'
    switch dataset
      % multi-class
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                % 88.26 / 82.09 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                % 92.83 / 83.01 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                % 89.64 / 82.03 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                % GPU ??? (non-smooth random kernels; batch size: 100; weight decay: 0.01)
    end
  case 'larpV1lP0+convV3P3+fcV1'
    switch dataset
      % multi-class
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                % 85.62 / 81.57 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                % 86.44 / 81.70 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                % 83.36 / 79.32 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                % GPU ??? (non-smooth random kernels; batch size: 100; weight decay: 0.01)
    end
  case 'larpV1lP1+convV3P3+fcV1'
    switch dataset
      % multi-class
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                % 76.37 / 73.58 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                % 79.02 / 74.63 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                % 76.30 / 72.92 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                % GPU ??? (non-smooth random kernels; batch size: 100; weight decay: 0.01)
    end
  case 'larpV2lP0+convV3P3+fcV1'
    switch dataset
      % multi-class
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                % 78.38 / 75.32 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                % 77.89 / 75.57 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                % 72.37 / 71.50 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                % GPU ??? (non-smooth random kernels; batch size: 100; weight decay: 0.01)
    end
  case 'larpV2lP1+convV3P3+fcV1'
    switch dataset
      % multi-class
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                % 68.63 / 65.06 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                % 68.86 / 66.89 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                % 62.53 / 61.19 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                % GPU ??? (non-smooth random kernels; batch size: 100; weight decay: 0.01)
    end
  case 'larpV2lP2+convV3P3+fcV1'
    switch dataset
      % multi-class
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                % 50.33 / 49.35 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                % 45.09 / 43.99 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                % 26.51 / 26.45 (non-smooth random kernels; batch size: 100; weight decay: 0.01)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                % GPU ??? (non-smooth random kernels; batch size: 100; weight decay: 0.01)
    end


%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%


  case 'TMP_NETWORK'
    switch dataset
      % multi-class
      case 'mnist-two-class-9-4'
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;
    end














  % case 'fc_lenet_with_larger_fc_conv'
  %   switch dataset
  %     % multi-class
  %     case 'mnist'
  %       learning_rate = [0.05*ones(1,15) 0.01*ones(1,20) 0.005*ones(1,10) 0.0005*ones(1,25)]; % testing!!!
  %     case 'mnist-multi-class-subsampled'
  %       learning_rate = [0.05*ones(1,15) 0.01*ones(1,20) 0.005*ones(1,10) 0.0005*ones(1,25)]; % testing!!!
  %   end
  % case 'lenet_with_larger_fc_conv'
  %   switch dataset
  %     % multi-class
  %     case 'mnist'
  %       learning_rate = [0.05*ones(1,15) 0.01*ones(1,20) 0.005*ones(1,10) 0.0005*ones(1,25)]; % best
  %     case 'mnist-multi-class-subsampled'
  %       learning_rate = [0.05*ones(1,15) 0.01*ones(1,20) 0.005*ones(1,10) 0.0005*ones(1,25)]; % best
  %   end



  % case 'lenet+1'
  %   switch dataset
  %     case 'cifar-two-class-deer-truck'
  %       learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)];
  %     case 'mnist-two-class-9-4'
  %       learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)];
  %     case 'svhn-two-class-9-4'
  %       learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)];
  %     case 'stl-10-two-class-airplane-bird'
  %       learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)] * 10;
  %     case 'stl-10-two-class-airplane-cat'
  %       learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)] * 10;
  %   end
  % case 'lenet++1'
  %   switch dataset
  %     case 'cifar-two-class-deer-truck'
  %       learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)];
  %     case 'mnist-two-class-9-4'
  %       learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)];
  %     case 'svhn-two-class-9-4'
  %       learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)];
  %     case 'stl-10-two-class-airplane-bird'
  %       learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)] * 10;
  %     case 'stl-10-two-class-airplane-cat'
  %       learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)] * 10;
  %   end
  % case 'alexnet'
  %   switch dataset
  %     case 'cifar'
  %       learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)]; % amir-LR
  %     case 'coil-100'
  %       % still testing ...
  %       learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)]; % amir-LR
  %     case 'mnist'
  %       % still testing ...
  %       learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)]; % amir-LR
  %     case 'stl-10'
  %       learning_rate = [0.5*ones(1,20) 0.05*ones(1,15)  0.1:-0.01:0.06 0.05*ones(1,10)];
  %     case 'svhn'
  %       learning_rate = [0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,10)];
  %     case 'cifar-two-class-deer-horse'
  %       % still testing ...
  %       learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,55)]; % amir-LR
  %     case 'cifar-two-class-deer-truck'
  %       % still testing ...
  %       learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,55)]; % amir-LR
  %     case 'mnist-two-class-9-4'
  %       % still testing ...
  %       learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,55)]; % amir-LR
  %     case 'svhn-two-class-9-4'
  %       % still testing ...
  %       learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,55)]; % amir-LR
  %     case 'prostate-v2-20-patients'
  %       % still testing ...
  %       learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)]; % amir-LR
  %     case 'prostate-v3-104-patients'
  %       % still testing ...
  %       learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)]; % amir-LR
  %   end
end




































