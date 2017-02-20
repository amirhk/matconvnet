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



  case 'cvv0p0+fcv1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 09.87 / 09.08 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 99.14 / 95.49 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 95.83 / 94.29 (bpd 0)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 18.92 / 19.59 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 18.92 / 19.59 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 83.83 / 77.09 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 83.08 / 77.80 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 76.94 / 73.69 (bpd 0)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 01.00  / 01.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 01.00  / 01.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 100.00 / 99.75 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 100.00 / 99.19 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 100.00 / 96.83 (bpd 0)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;       % 99.74 / 42.96 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;       % 95.26 / 43.70 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 69.34 / 44.08 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 48.26 / 41.20 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 38.26 / 36.29 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 30.38 / 30.13 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 24.12 / 23.89 (bpd 0)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 87.39 / 37.74 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 60.91 / 36.43 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 24.08 / 22.48 (bpd 0)
    end


  case 'cvv1p0+fcv1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 09.87 / 09.80 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 99.29 / 96.96 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 99.75 / 96.91 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 98.08 / 96.48 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 94.92 / 94.06 (bpd 0)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 69.94 / 62.57 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 85.41 / 74.64 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 84.63 / 75.38 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 79.83 / 73.92 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 70.95 / 67.25 (bpd 0)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 04.00  / 04.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 100.00 / 99.75 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 100.00 / 99.19 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 99.53  / 97.69 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 90.83  / 89.75 (bpd 0)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;       % 99.06 / 50.43 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;       % 72.84 / 50.13 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 48.64 / 43.51 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 35.46 / 35.05 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 26.54 / 26.29 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 18.90 / 19.18 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 12.22 / 12.28 (bpd 0)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 37.30 / 29.05 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 79.02 / 48.28 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 67.75 / 45.66 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 44.71 / 39.64 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 29.28 / 27.79 (bpd 0)
    end
  case 'cvv1p1+fcv1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 09.87 / 09.80 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 95.48 / 94.73 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 99.34 / 97.80 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 98.29 / 97.26 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 96.40 / 95.94 (bpd 0)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 81.19 / 74.16 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 88.21 / 78.70 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 86.33 / 78.54 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 82.20 / 77.29 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 75.59 / 72.12 (bpd 0)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 32.19  / 31.92 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 100.00 / 99.92 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 100.00 / 99.75 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 99.61  / 98.94 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 90.42  / 89.81 (bpd 0)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;       % 88.60 / 56.99 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;       % 60.00 / 52.66 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 44.78 / 42.95 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 31.68 / 31.81 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 25.36 / 24.66 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 18.18 / 18.08 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 12.72 / 13.05 (bpd 0)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 61.46 / 51.37 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 66.64 / 57.18 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 57.13 / 52.99 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 44.28 / 42.87 (bpd 0)
    end


  case 'cvv3p0+fcv1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 20.99 / 21.08 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 99.45 / 96.93 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 99.52 / 96.57 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 97.48 / 96.01 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 93.82 / 93.46 (bpd 0)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 57.77 / 52.74 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 86.64 / 73.43 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 83.06 / 73.67 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 76.26 / 70.95 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 66.15 / 62.87 (bpd 0)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 03.00 / 03.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 89.86 / 88.22 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 99.69 / 98.58 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 98.03 / 95.83 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 84.28 / 82.75 (bpd 0)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;       % 98.66 / 48.68 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;       % 66.64 / 48.11 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 43.76 / 40.75 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 33.80 / 33.26 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 22.74 / 22.66 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 16.78 / 16.41 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 13.16 / 12.60 (bpd 0)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 69.61 / 39.71 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 56.87 / 41.42 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 37.90 / 34.28 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 27.99 / 27.56 (bpd 0)
    end
  case 'cvv3p1+fcv1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 21.01 / 21.06 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 99.61 / 98.23 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 99.52 / 98.08 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 98.31 / 97.28 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 95.96 / 95.71 (bpd 0)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 85.75 / 74.86 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 88.66 / 76.79 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 84.04 / 76.10 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 77.82 / 73.11 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 69.26 / 66.03 (bpd 0)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 91.08  / 90.47 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 100.00 / 99.94 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 99.97  / 99.81 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 98.36  / 98.42 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 85.44  / 85.19 (bpd 0)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;       % 65.48 / 49.62 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;       % 44.80 / 42.06 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 33.70 / 33.73 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 24.50 / 24.69 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 17.98 / 18.31 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 16.90 / 16.46 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 12.64 / 12.09 (bpd 0)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 50.50 / 46.50 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 78.63 / 58.58 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 66.27 / 57.26 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 55.29 / 51.97 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 44.25 / 43.13 (bpd 0)
    end
  case 'cvv3p3+fcv1' % = lenet_bu
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 09.87 / 09.80 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 97.92 / 97.44 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 97.67 / 97.47 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 95.85 / 95.70 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 91.87 / 92.50 (bpd 0)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 75.16 / 70.83 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 75.32 / 71.80 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 71.10 / 68.43 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 63.95 / 63.03 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 50.66 / 50.94 (bpd 0)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 99.97 / 100.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 99.69 / 99.72 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 96.00 / 95.94 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 85.19 / 85.92 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 34.72 / 34.97 (bpd 0)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;       % 37.34 / 38.05 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;       % 26.68 / 26.85 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 17.02 / 17.06 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 15.30 / 14.68 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 14.80 / 14.85 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 11.26 / 11.29 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 11.16 / 11.14 (bpd 0)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 48.17 / 47.63 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 58.42 / 57.58 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 57.27 / 56.37 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 52.91 / 52.86 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 45.91 / 46.11 (bpd 0)


      % case 'stl-10'
      %   % learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];                                        % 15.05 / 14.69 (bpd 0)
      %   % learning_rate = [0.5*ones(1,20) 0.05*ones(1,15)  0.1:-0.01:0.06 0.05*ones(1,10)];                            % 27.15 / 27.48 (bpd 0)
      %   % learning_rate = [0.5*ones(1,50) 0.05*ones(1,100)];                                                           % 36.10 / 36.60 (bpd 0)
      %   % learning_rate = [0.5*ones(1,150)];                                                                           % 45.32 / 44.85 (bpd 0)
      %   learning_rate = [1 * ones(1,20) 0.5*ones(1,130)];                                                              % 46.48 / 45.55 (bpd 0)
    end


  case 'cvv5hp0+fcv1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 09.87 / 09.80 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 20.94 / 21.04 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 92.47 / 91.24 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 91.16 / 90.94 (bpd 0) % BEST
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 18.92 / 19.60 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 18.92 / 19.59 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 32.30 / 32.68 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 58.14 / 52.84 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 55.48 / 52.22 (bpd 0) % BEST
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 01.00 / 01.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 01.00 / 01.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 01.00 / 01.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 01.75 / 02.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 03.28 / 03.78 (bpd 0) % BEST
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;       % 31.06 / 22.64 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;       % 93.58 / 46.88 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 68.34 / 46.02 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 46.10 / 40.41 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 34.30 / 33.63 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 24.28 / 24.29 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 17.14 / 17.14 (bpd 0)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 23.54 / 23.80 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 24.75 / 23.84 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 24.00 / 23.46 (bpd 0) % BEST
    end
  case 'cvv5hp1+fcv1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 09.87 / 09.80 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 09.87 / 09.80 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 20.98 / 21.05 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 97.61 / 96.28 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 95.61 / 95.50 (bpd 0) % BEST
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 18.92 / 19.59 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 18.93 / 19.59 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 72.87 / 66.80 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 71.54 / 66.19 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 63.56 / 60.31 (bpd 0)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 01.00 / 01.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 01.00 / 01.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 88.25 / 88.94 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 81.28 / 81.53 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 24.11 / 25.06 (bpd 0)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;       % 61.56 / 46.10 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;       % 61.94 / 48.68 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 47.10 / 43.06 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 36.66 / 35.38 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 24.54 / 24.09 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 13.94 / 14.20 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 11.88 / 11.54 (bpd 0)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 24.56 / 25.05 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 54.84 / 51.61 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 44.37 / 43.39 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 37.34 / 37.11 (bpd 0)
    end
  case 'cvv5hp3+fcv1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 09.87 / 09.80 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 20.85 / 20.98 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 95.69 / 95.72 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 94.47 / 94.90 (bpd 0) % BEST
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 18.92 / 19.59 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 18.92 / 19.59 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 68.53 / 65.04 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 68.20 / 66.11 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 59.22 / 58.56 (bpd 0)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 01.00 / 01.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 80.86 / 81.36 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 96.14 / 96.39 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 89.36 / 90.14 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 66.97 / 68.33 (bpd 0)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;       % 45.70 / 44.10 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;       % 37.00 / 37.11 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 27.76 / 28.14 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 16.42 / 15.68 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 11.34 / 10.64 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 10.00 / 10.00 (bpd 0)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 10.83 / 10.94 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 47.67 / 47.03 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 51.75 / 51.35 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 49.61 / 48.91 (bpd 0)
    end
  case 'cvv5hp5+fcv1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 84.92 / 85.32 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 84.82 / 84.95 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 81.72 / 82.17 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 75.67 / 76.83 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 61.69 / 62.68 (bpd 0)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 37.39 / 35.55 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 34.30 / 33.98 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 26.94 / 28.63 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 23.42 / 24.74 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 18.94 / 19.63 (bpd 0)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 83.97 / 85.17 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 61.14 / 61.47 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 11.22 / 10.81 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 04.58 / 04.67 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 01.19 / 01.36 (bpd 0)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;       % 10.64 / 10.95 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;       % 13.58 / 13.58 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 11.82 / 11.76 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 12.38 / 12.51 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 10.00 / 10.00 (bpd 0)
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 38.51 / 38.75 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 39.41 / 39.83 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 36.46 / 36.85 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 31.00 / 30.36 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 20.12 / 19.77 (bpd 0)
    end


  case 'cvv5ap0+fcv1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 91.40 / 91.64 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 82.74 / 83.66 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 55.05 / 55.21 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 22.44 / 22.52 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 19.75 / 20.43 (bpd 0)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 36.03 / 36.66 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 18.92 / 19.59 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 18.92 / 19.59 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 18.92 / 19.59 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 18.92 / 19.59 (bpd 0)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 04.06 / 03.92 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 02.25 / 02.08 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 01.47 / 01.39 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 01.03 / 00.83 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 00.97 / 00.89 (bpd 0)
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;       % << didn't test >> (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;       % 08.68 / 08.44 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 08.38 / 08.43 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 08.36 / 08.24 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 08.30 / 08.26 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 08.08 / 07.78 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 08.14 / 07.93 (bpd 0) % BEST
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 26.85 / 26.62 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 16.87 / 16.63 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 11.34 / 11.35 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 10.53 / 10.13 (bpd 0)
    end
  case 'cvv5ap1+fcv1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 89.95 / 90.71 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 79.88 / 80.79 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 40.64 / 40.85 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 26.55 / 27.02 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 25.55 / 25.49 (bpd 0)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 18.92 / 19.59 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 18.92 / 19.59 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 18.92 / 19.59 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 18.92 / 19.59 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 18.79 / 19.44 (bpd 0) % BEST
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 01.89 / 01.86 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 01.58 / 01.08 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 01.47 / 00.11 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 01.06 / 00.69 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 00.67 / 01.08 (bpd 0) % BEST
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;       % << didn't test >> (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;       % 11.18 / 11.58 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 10.94 / 11.24 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 10.48 / 10.63 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 10.90 / 11.14 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 10.76 / 10.69 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 10.58 / 10.59 (bpd 0) % BEST
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 26.37 / 26.40 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 16.01 / 15.88 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 10.00 / 10.00 (bpd 0)
    end
  case 'cvv5ap3+fcv1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 76.44 / 77.97 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 31.51 / 31.18 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 11.06 / 11.11 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 14.79 / 15.46 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 09.85 / 10.08 (bpd 0)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 18.92 / 19.58 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 18.92 / 19.59 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 18.92 / 19.59 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 18.92 / 19.59 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 13.13 / 13.73 (bpd 0)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 01.00 / 01.00 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 01.03 / 01.06 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 00.92 / 00.97 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 00.97 / 00.89 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 01.00 / 00.89 (bpd 0) % BEST
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;       % << didn't test >> (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;       % 09.20 / 09.16 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 09.36 / 09.13 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 09.30 / 09.14 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 09.46 / 09.13 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 09.36 / 09.23 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 09.32 / 09.23 (bpd 0) % BEST
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 15.49 / 15.32 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 10.18 / 10.13 (bpd 0)
    end
  case 'cvv5ap5+fcv1'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 16.14 / 16.41 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 14.62 / 14.30 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 10.31 / 10.07 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 12.15 / 12.34 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 10.10 / 10.13 (bpd 0)
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 18.92 / 19.59 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 18.92 / 19.59 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 18.92 / 19.59 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 14.69 / 16.24 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 13.98 / 15.23 (bpd 0)
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 01.00 / 01.00 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 01.00 / 01.00 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 00.92 / 00.94 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 00.94 / 00.97 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 00.86 / 00.94 (bpd 0) % BEST
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;       % << didn't test >> (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;       % 11.20 / 11.06 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 10.94 / 11.01 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 11.22 / 11.08 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 11.38 / 10.96 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 11.26 / 10.91 (bpd 0) % BEST
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 11.08 / 11.14 (bpd 0) % BEST
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;       % 10.00 / 10.00 (bpd 0)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;       % 11.18 / 11.28 (bpd 0) % BEST
    end


%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%


    case 'cvv0p0+fcv2'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;                                         % --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;                                         % --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
    end


  case 'cvv1p0+fcv2'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;                                         % --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;                                         % --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
    end
  case 'cvv1p1+fcv2'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;                                         % --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;                                         % --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
    end


  case 'cvv3p0+fcv2'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;                                         % --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;                                         % --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
    end
  case 'cvv3p1+fcv2'
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;                                         % --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;                                         % --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
    end
  case 'cvv3p3+fcv2' % = lenet_bu
    switch dataset
      % multi-class
      case 'mnist'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'svhn'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'coil-100'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'stl-10'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 010;                                         % --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] * 003;                                         % --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                         % GPU 1 --.-- / --.--
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                         % GPU 2 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                         % GPU 3 --.-- / --.--
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                         % GPU 4 --.-- / --.--
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




































