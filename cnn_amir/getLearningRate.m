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
  case 'lenet'
    switch dataset
      % multi-class
      case 'cifar'
        learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
      case 'coil-100'
        learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
      case 'mnist'
        learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
      case 'stl-10'
        learning_rate = [0.5*ones(1,20) 0.05*ones(1,15)  0.1:-0.01:0.06 0.05*ones(1,10)];
      case 'svhn'
        learning_rate = [0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,10)];
      case 'cifar-two-class-deer-horse'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,120)];
      case 'cifar-two-class-deer-truck'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,120)];
      case 'cifar-no-white-two-class-deer-truck'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,120)];

      % multi-class subsampled
      case 'mnist-multi-class-subsampled'
        learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
      case 'svhn-multi-class-subsampled'
        learning_rate = [0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,10)];
      case 'cifar-multi-class-subsampled'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,25)];
      case 'stl-10-multi-class-subsampled'
        learning_rate = [0.5*ones(1,20) 0.05*ones(1,15)  0.1:-0.01:0.06 0.05*ones(1,10)];

      % two-class
      case 'mnist-two-class-9-4'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,120)];
      case 'stl-10-two-class-airplane-bird'
        learning_rate = [0.05*ones(1,20) 0.005*ones(1,20) 0.001*ones(1,110)] * 10;
      case 'stl-10-two-class-airplane-cat'
        learning_rate = [0.05*ones(1,20) 0.005*ones(1,20) 0.001*ones(1,110)] * 10;
        % learning_rate = [0.05*ones(1,20) 0.005*ones(1,20) 0.001*ones(1,100) 0.0001*ones(1,310)] * 10;
      case 'svhn-two-class-9-4'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,120)];
      case 'prostate-v2-20-patients'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20)];
      case 'prostate-v3-104-patients'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20)];
    end



  case 'cvv0p0+fcv1'
    switch dataset
      % multi-class
      case 'mnist'
        learning_rate = [0.1*ones(1,5) 0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,45)] / 20;
      case 'cifar'
        % learning_rate = [0.1*ones(1,5) 0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,45)] / 20;               % 49.79 / 37.38
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                % 10.00 / 10.00
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                % 10.00 / 10.00
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                % 87.39 / 37.74 # BEST
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                % 60.91 / 36.43
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;                                % 24.08 / 22.48
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 300;                                % 15.61 / 15.23
      case 'stl-10'
        learning_rate = [0.1*ones(1,5) 0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,45)] / 20;
      case 'svhn'
        learning_rate = [0.1*ones(1,5) 0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,45)] / 20;

      % multi-class subsampled
      % case 'mnist-multi-class-subsampled'
      %   learning_rate = [0.1*ones(1,5) 0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,45)] / 20;
      % case 'cifar-multi-class-subsampled'
      %   learning_rate = [0.1*ones(1,5) 0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,45)] / 20;
      % case 'stl-10-multi-class-subsampled'
      %   learning_rate = [0.1*ones(1,5) 0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,45)] / 20;
      % case 'svhn-multi-class-subsampled'
        learning_rate = [0.1*ones(1,5) 0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,45)] / 20;
    end
  case 'cvv3p0+fcv1'
    switch dataset
      % multi-class
      case 'cifar'
        % learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];                                  % 35.89 / 29.74 (bpd 0 / 3)
        % learning_rate = [0.1 * ones(1,25) 0.05*ones(1,50)];                                                    % 10.00 / 10.00 (bpd 0 / 3)
        learning_rate = [0.01*ones(1,25) 0.005*ones(1,25) 0.001*ones(1,25)];                                     % 60.24 / 41.92 (bpd 0 / 3) % BEST!!!
        % learning_rate = [0.01*ones(1,25) 0.005*ones(1,25) 0.0001*ones(1,25)];                                  % 56.34 / 40.22 (bpd 0 / 3)
        % learning_rate = [0.01*ones(1,10) 0.005*ones(1,20) 0.0001*ones(1,20) 0.00005*ones(1,25)];               % 43.82 / 37.85 (bpd 0 / 3)
        % learning_rate = [0.01*ones(1,10) 0.005*ones(1,20) 0.0001*ones(1,20) 0.01*ones(1,5) 0.0001*ones(1,20)]; % 46.83 / 37.94 (bpd 0 / 3)
    end
  case 'cvv3p3+fcv1' % = lenet_bu
    switch dataset
      % multi-class (all of the dataset)
      case 'mnist'
        learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
      case 'cifar'
        learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];                                    % 54.54 / 54.07
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                                % GPU 3
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                                % GPU 4
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                                % GPU 1
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                % GPU ???
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;                                % GPU ???
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 300;                                % GPU ???
      case 'stl-10'
        % learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];             % 15.05 / 14.69 (bpd 0 / 3)
        % learning_rate = [0.5*ones(1,20) 0.05*ones(1,15)  0.1:-0.01:0.06 0.05*ones(1,10)]; % 27.15 / 27.48 (bpd 0 / 3)
        % learning_rate = [0.5*ones(1,50) 0.05*ones(1,100)];                                % 36.10 / 36.60 (bpd 0 / 3)
        % learning_rate = [0.5*ones(1,150)];                                                % 45.32 / 44.85 (bpd 0 / 3)
        learning_rate = [1 * ones(1,20) 0.5*ones(1,130)];                                   % 46.48 / 45.55 (bpd 0 / 3)
      case 'svhn'
        % learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];                                % 72.80 / 70.39 (bpd 0 / 3)
        % learning_rate = [0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,10)];                                % 75.10 / 72.01 (bpd 0 / 3)
        % learning_rate = [0.05*ones(1,150)];                                                                  % 76.75 / 71.82 (bpd 0 / 3)
        learning_rate = [0.05*ones(1,50) 0.01*ones(1,50) 0.005*ones(1,50)];                                    % 78.37 / 74.12 (bpd 0 / 3)

      % multi-class subsampled % TODO!!!!!!!!!! should follow above
      % case 'mnist-multi-class-subsampled'
      %   learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
      % case 'cifar-multi-class-subsampled'
      %   learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
      % case 'stl-10-multi-class-subsampled'
      %   learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
      % case 'svhn-multi-class-subsampled'
        learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
    end
  case 'cvv5p0+fcv1'
    switch dataset
      % multi-class
      case 'cifar'
        % learning_rate = [0.1*ones(1,75)];                                                                      % 10.00 / 10.00 (bpd 0 / 5)
        % learning_rate = [0.05*ones(1,75)];                                                                     % 10.00 / 10.00 (bpd 0 / 5)
        % learning_rate = [0.01*ones(1,75)];                                                                     % 10.00 / 10.00 (bpd 0 / 5)
        learning_rate = [0.005*ones(1,75)];                                                                      % 27.66 / 25.49 (bpd 0 / 5) % BEST!!! (considering final perf & number of epochs)
        % learning_rate = [0.001*ones(1,75)];                                                                    % 25.79 / 24.86 (bpd 0 / 5)
        % learning_rate = [0.0005*ones(1,150)];                                                                  % 27.46 / 26.04 (bpd 0 / 5)
        % learning_rate = [0.0001*ones(1,150)];                                                                  % 23.63 / 23.86 (bpd 0 / 5)
        % learning_rate = [0.01*ones(1,10) 0.005*ones(1,90) 0.001*ones(1,50)];                                   % 26.96 / 21.12 (bpd 0 / 5)
        % learning_rate = [0.01*ones(1,75) 0.001*ones(1,75)];                                                    % 24.30 / 23.17 (bpd 0 / 5)
    end
  case 'cvv5p3+fcv1'
    switch dataset
      % multi-class
      case 'cifar'
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 001;                              % 10.00 / 10.00 (bpd 0 / 5)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 003;                              % 10.83 / 10.94 (bpd 0 / 5)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 010;                              % 47.67 / 47.03 (bpd 0 / 5)
        learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 030;                                % 51.75 / 51.35 (bpd 0 / 5) % BEST!!!
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 100;                              % 49.61 / 48.91 (bpd 0 / 5)
        % learning_rate = [0.1 * ones(1,25) 0.03*ones(1,25) 0.01*ones(1,25)] / 300;                              % 44.07 / 43.93 (bpd 0 / 5)
    end





  case 'fc_lenet_with_larger_fc_conv'
    switch dataset
      % multi-class
      case 'mnist'
        learning_rate = [0.05*ones(1,15) 0.01*ones(1,20) 0.005*ones(1,10) 0.0005*ones(1,25)]; % testing!!!
      case 'mnist-multi-class-subsampled'
        learning_rate = [0.05*ones(1,15) 0.01*ones(1,20) 0.005*ones(1,10) 0.0005*ones(1,25)]; % testing!!!
    end
  case 'lenet_with_larger_fc_conv'
    switch dataset
      % multi-class
      case 'mnist'
        learning_rate = [0.05*ones(1,15) 0.01*ones(1,20) 0.005*ones(1,10) 0.0005*ones(1,25)]; % best
      case 'mnist-multi-class-subsampled'
        learning_rate = [0.05*ones(1,15) 0.01*ones(1,20) 0.005*ones(1,10) 0.0005*ones(1,25)]; % best
    end





  case 'lenet+1'
    switch dataset
      case 'cifar-two-class-deer-truck'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)];
      case 'mnist-two-class-9-4'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)];
      case 'svhn-two-class-9-4'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)];
      case 'stl-10-two-class-airplane-bird'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)] * 10;
      case 'stl-10-two-class-airplane-cat'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)] * 10;
    end
  case 'lenet++1'
    switch dataset
      case 'cifar-two-class-deer-truck'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)];
      case 'mnist-two-class-9-4'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)];
      case 'svhn-two-class-9-4'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)];
      case 'stl-10-two-class-airplane-bird'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)] * 10;
      case 'stl-10-two-class-airplane-cat'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)] * 10;
    end
  case 'alexnet'
    switch dataset
      case 'cifar'
        learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)]; % amir-LR
      case 'coil-100'
        % still testing ...
        learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)]; % amir-LR
      case 'mnist'
        % still testing ...
        learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)]; % amir-LR
      case 'stl-10'
        learning_rate = [0.5*ones(1,20) 0.05*ones(1,15)  0.1:-0.01:0.06 0.05*ones(1,10)];
      case 'svhn'
        learning_rate = [0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,10)];
      case 'cifar-two-class-deer-horse'
        % still testing ...
        learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,55)]; % amir-LR
      case 'cifar-two-class-deer-truck'
        % still testing ...
        learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,55)]; % amir-LR
      case 'mnist-two-class-9-4'
        % still testing ...
        learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,55)]; % amir-LR
      case 'svhn-two-class-9-4'
        % still testing ...
        learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,55)]; % amir-LR
      case 'prostate-v2-20-patients'
        % still testing ...
        learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)]; % amir-LR
      case 'prostate-v3-104-patients'
        % still testing ...
        learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)]; % amir-LR
    end
end




































