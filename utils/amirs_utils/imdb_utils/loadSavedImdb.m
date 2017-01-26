% -------------------------------------------------------------------------
function imdb = loadSavedImdb(input_opts)
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

  dataset = getValueFromFieldOrDefault(input_opts, 'dataset', 'mnist-two-class-9-4');
  network_arch = getValueFromFieldOrDefault(input_opts, 'network_arch', 'lenet');
  posneg_balance = getValueFromFieldOrDefault(input_opts, 'posneg_balance', 'balanced-low');
  fold_number = getValueFromFieldOrDefault(input_opts, 'fold_number', 1); % currently only implemented for prostate data

  if ~isTwoClassImdb(dataset)
    afprintf(sprintf('[INFO] Loading all-class imdb (dataset: %s, network_arch: %s)\n', dataset, network_arch));
    imdb = load(fullfile(getDevPath(), 'data', 'imdb', sprintf('%s-%s', dataset, network_arch), 'imdb.mat'));
  else
    afprintf(sprintf('[INFO] Loading two-class imdb (dataset: %s, posneg_balance: %s)\n', dataset, posneg_balance));
    path_to_imdbs = fullfile(getDevPath(), 'data', 'two_class_imdbs');
    switch dataset
      case 'mnist-two-class-9-4'
        % currently fold number is not implemented.
        switch posneg_balance
          case 'balanced-low'
            tmp = load(fullfile(path_to_imdbs, 'mnist', 'saved-two-class-mnist-pos9-neg4-balanced-low-train-29-29.mat'));
          case 'unbalanced'
            tmp = load(fullfile(path_to_imdbs, 'mnist', 'saved-two-class-mnist-pos9-neg4-unbalanced-train-29-6131.mat'));
          case 'balanced-high'
            tmp = load(fullfile(path_to_imdbs, 'mnist', 'saved-two-class-mnist-pos9-neg4-balanced-high-train-5851-6131.mat'));
        end
      case 'cifar-two-class-deer-horse'
        % currently fold number is not implemented.
        switch posneg_balance
          case 'balanced-low'
            tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-two-class-cifar-pos5-neg8-balanced-low-train-25-25.mat'));
          case 'unbalanced'
            tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-two-class-cifar-pos5-neg8-unbalanced-train-25-5000.mat'));
          case 'balanced-high'
            tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-two-class-cifar-pos5-neg8-balanced-high-train-5000-5000.mat'));
        end
      case 'cifar-two-class-deer-truck'
        % currently fold number is not implemented.
        switch posneg_balance
          case 'balanced-low'
            tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-two-class-cifar-pos5-neg10-balanced-low-train-25-25.mat'));
          case 'unbalanced'
            tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-two-class-cifar-pos5-neg10-unbalanced-train-25-5000.mat'));
          case 'balanced-high'
            tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-two-class-cifar-pos5-neg10-balanced-high-train-5000-5000.mat'));
        end
      case 'svhn-two-class-9-4'
        % currently fold number is not implemented.
        switch posneg_balance
          case 'balanced-low'
            tmp = load(fullfile(path_to_imdbs, 'svhn', 'saved-two-class-svhn-pos9-neg4-balanced-low-train-23-23.mat'));
          case 'unbalanced'
            tmp = load(fullfile(path_to_imdbs, 'svhn', 'saved-two-class-svhn-pos9-neg4-unbalanced-train-23-7458.mat'));
          case 'balanced-high'
            tmp = load(fullfile(path_to_imdbs, 'svhn', 'saved-two-class-svhn-pos9-neg4-balanced-high-train-4659-7458.mat'));
        end
      case 'prostate-v2-20-patients'
        switch posneg_balance
          case 'k=5-fold-unbalanced'
            switch fold_number
              case 1
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-unbalanced-train-51-655.mat'));
              case 2
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-unbalanced-train-62-597.mat'));
              case 3
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-unbalanced-train-68-544.mat'));
              case 4
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-unbalanced-train-68-567.mat'));
              case 5
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-unbalanced-train-71-493.mat'));
            end
          case 'k=5-fold-balanced-high'
            switch fold_number
              case 1
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-balanced-high-train-488-597.mat'));
              case 2
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-balanced-high-train-504-498.mat'));
              case 3
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-balanced-high-train-512-574.mat'));
              case 4
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-balanced-high-train-512-599.mat'));
              case 5
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-balanced-high-train-544-588.mat'));
            end
          case 'leave-one-out-unbalanced'
            switch fold_number
              case 1
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-01-unbalanced-train-77-677.mat'));
              case 2
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-02-unbalanced-train-77-659.mat'));
              case 3
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-03-unbalanced-train-75-669.mat'));
              case 4
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-04-unbalanced-train-76-691.mat'));
              case 5
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-05-unbalanced-train-70-692.mat'));
              case 6
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-06-unbalanced-train-80-636.mat'));
              case 7
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-07-unbalanced-train-80-692.mat'));
              case 8
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-08-unbalanced-train-78-691.mat'));
              case 9
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-09-unbalanced-train-75-678.mat'));
              case 10
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-10-unbalanced-train-74-688.mat'));
              case 11
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-11-unbalanced-train-75-701.mat'));
              case 12
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-12-unbalanced-train-77-696.mat'));
              case 13
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-13-unbalanced-train-71-703.mat'));
              case 14
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-14-unbalanced-train-78-603.mat'));
              case 15
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-15-unbalanced-train-79-671.mat'));
              case 16
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-16-unbalanced-train-77-677.mat'));
              case 17
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-17-unbalanced-train-77-691.mat'));
              case 18
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-18-unbalanced-train-75-660.mat'));
              case 19
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-19-unbalanced-train-80-681.mat'));
              case 20
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-20-unbalanced-train-69-710.mat'));
            end
          case 'leave-one-out-balanced-high'
            switch fold_number
              case 1
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-01-balanced-high-train-568-703.mat'));
              case 2
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-02-balanced-high-train-600-669.mat'));
              case 3
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-03-balanced-high-train-592-688.mat'));
              case 4
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-04-balanced-high-train-608-691.mat'));
              case 5
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-05-balanced-high-train-616-659.mat'));
              case 6
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-06-balanced-high-train-560-692.mat'));
              case 7
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-07-balanced-high-train-624-691.mat'));
              case 8
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-08-balanced-high-train-640-636.mat'));
              case 9
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-09-balanced-high-train-600-660.mat'));
              case 10
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-10-balanced-high-train-624-603.mat'));
              case 11
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-11-balanced-high-train-600-701.mat'));
              case 12
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-12-balanced-high-train-640-681.mat'));
              case 13
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-13-balanced-high-train-616-677.mat'));
              case 14
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-14-balanced-high-train-552-710.mat'));
              case 15
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-15-balanced-high-train-616-696.mat'));
              case 16
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-16-balanced-high-train-616-691.mat'));
              case 17
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-17-balanced-high-train-640-692.mat'));
              case 18
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-18-balanced-high-train-600-678.mat'));
              case 19
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-19-balanced-high-train-632-671.mat'));
              case 20
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-20-balanced-high-train-616-677.mat'));
            end
          case 'leave-one-out-balanced-640-640'
            switch fold_number
              case 1
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-01-balanced-640-640-train-616-616.mat'));
              case 2
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-02-balanced-640-640-train-624-624.mat'));
              case 3
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-03-balanced-640-640-train-616-616.mat'));
              case 4
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-04-balanced-640-640-train-608-608.mat'));
              case 5
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-05-balanced-640-640-train-624-624.mat'));
              case 6
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-06-balanced-640-640-train-616-616.mat'));
              case 7
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-07-balanced-640-640-train-640-640.mat'));
              case 8
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-08-balanced-640-640-train-560-560.mat'));
              case 9
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-09-balanced-640-640-train-568-568.mat'));
              case 10
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-10-balanced-640-640-train-616-616.mat'));
              case 11
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-11-balanced-640-640-train-616-616.mat'));
              case 12
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-12-balanced-640-640-train-552-552.mat'));
              case 13
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-13-balanced-640-640-train-600-600.mat'));
              case 14
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-14-balanced-640-640-train-640-640.mat'));
              case 15
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-15-balanced-640-640-train-600-600.mat'));
              case 16
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-16-balanced-640-640-train-640-640.mat'));
              case 17
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-17-balanced-640-640-train-600-600.mat'));
              case 18
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-18-balanced-640-640-train-592-592.mat'));
              case 19
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-19-balanced-640-640-train-600-600.mat'));
              case 20
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-20-balanced-640-640-train-632-632.mat'));
            end
          case 'leave-one-out-balanced-1280-1280'
            switch fold_number
              case 1
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-01-balanced-1280-1280-train-1216-1216.mat'));
              case 2
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-02-balanced-1280-1280-train-1200-1200.mat'));
              case 3
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-03-balanced-1280-1280-train-1232-1232.mat'));
              case 4
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-04-balanced-1280-1280-train-1232-1232.mat'));
              case 5
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-05-balanced-1280-1280-train-1264-1264.mat'));
              case 6
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-06-balanced-1280-1280-train-1136-1136.mat'));
              case 7
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-07-balanced-1280-1280-train-1200-1200.mat'));
              case 8
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-08-balanced-1280-1280-train-1280-1280.mat'));
              case 9
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-09-balanced-1280-1280-train-1280-1280.mat'));
              case 10
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-10-balanced-1280-1280-train-1120-1120.mat'));
              case 11
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-11-balanced-1280-1280-train-1248-1248.mat'));
              case 12
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-12-balanced-1280-1280-train-1200-1200.mat'));
              case 13
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-13-balanced-1280-1280-train-1104-1104.mat'));
              case 14
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-14-balanced-1280-1280-train-1232-1232.mat'));
              case 15
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-15-balanced-1280-1280-train-1232-1232.mat'));
              case 16
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-16-balanced-1280-1280-train-1200-1200.mat'));
              case 17
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-17-balanced-1280-1280-train-1232-1232.mat'));
              case 18
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-18-balanced-1280-1280-train-1184-1184.mat'));
              case 19
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-19-balanced-1280-1280-train-1248-1248.mat'));
              case 20
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-20-balanced-1280-1280-train-1280-1280.mat'));
            end
        end
      case 'prostate-v3-104-patients'
        switch posneg_balance
          case 'leave-one-out-balanced-low'
            switch fold_number
              case 01
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-01-balanced-low-train-131-131.mat'));
              case 02
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-02-balanced-low-train-132-132.mat'));
              case 03
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-03-balanced-low-train-133-133.mat'));
              case 04
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-04-balanced-low-train-133-133.mat'));
              case 05
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-05-balanced-low-train-133-133.mat'));
              case 06
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-06-balanced-low-train-133-133.mat'));
              case 07
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-07-balanced-low-train-133-133.mat'));
              case 08
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-08-balanced-low-train-133-133.mat'));
              case 09
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-09-balanced-low-train-129-129.mat'));
              case 10
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-10-balanced-low-train-133-133.mat'));
              case 11
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-11-balanced-low-train-133-133.mat'));
              case 12
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-12-balanced-low-train-133-133.mat'));
              case 13
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-13-balanced-low-train-133-133.mat'));
              case 14
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-14-balanced-low-train-133-133.mat'));
              case 15
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-15-balanced-low-train-133-133.mat'));
              case 16
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-16-balanced-low-train-133-133.mat'));
              case 17
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-17-balanced-low-train-133-133.mat'));
              case 18
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-18-balanced-low-train-133-133.mat'));
              case 19
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-19-balanced-low-train-133-133.mat'));
              case 20
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-20-balanced-low-train-133-133.mat'));
              case 21
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-21-balanced-low-train-133-133.mat'));
              case 22
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-22-balanced-low-train-133-133.mat'));
              case 23
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-23-balanced-low-train-133-133.mat'));
              case 24
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-24-balanced-low-train-133-133.mat'));
              case 25
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-25-balanced-low-train-133-133.mat'));
              case 26
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-26-balanced-low-train-133-133.mat'));
              case 27
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-27-balanced-low-train-133-133.mat'));
              case 28
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-28-balanced-low-train-131-131.mat'));
              case 29
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-29-balanced-low-train-128-128.mat'));
              case 30
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-30-balanced-low-train-133-133.mat'));
              case 31
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-31-balanced-low-train-130-130.mat'));
              case 32
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-32-balanced-low-train-133-133.mat'));
              case 33
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-33-balanced-low-train-133-133.mat'));
              case 34
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-34-balanced-low-train-133-133.mat'));
              case 35
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-35-balanced-low-train-133-133.mat'));
              case 36
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-36-balanced-low-train-133-133.mat'));
              case 37
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-37-balanced-low-train-133-133.mat'));
              case 38
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-38-balanced-low-train-133-133.mat'));
              case 39
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-39-balanced-low-train-129-129.mat'));
              case 40
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-40-balanced-low-train-133-133.mat'));
              case 41
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-41-balanced-low-train-133-133.mat'));
              case 42
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-42-balanced-low-train-131-131.mat'));
              case 43
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-43-balanced-low-train-133-133.mat'));
              case 44
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-44-balanced-low-train-133-133.mat'));
              case 45
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-45-balanced-low-train-133-133.mat'));
              case 46
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-46-balanced-low-train-130-130.mat'));
              case 47
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-47-balanced-low-train-132-132.mat'));
              case 48
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-48-balanced-low-train-133-133.mat'));
              case 49
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-49-balanced-low-train-130-130.mat'));
              case 50
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-50-balanced-low-train-126-126.mat'));
              case 51
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-51-balanced-low-train-132-132.mat'));
              case 52
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-52-balanced-low-train-133-133.mat'));
              case 53
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-53-balanced-low-train-125-125.mat'));
              case 54
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-54-balanced-low-train-133-133.mat'));
              case 55
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-55-balanced-low-train-133-133.mat'));
              case 56
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-56-balanced-low-train-130-130.mat'));
              case 57
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-57-balanced-low-train-133-133.mat'));
              case 58
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-58-balanced-low-train-133-133.mat'));
              case 59
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-59-balanced-low-train-131-131.mat'));
              case 60
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-60-balanced-low-train-133-133.mat'));
              case 61
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-61-balanced-low-train-133-133.mat'));
              case 62
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-62-balanced-low-train-126-126.mat'));
              case 63
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-63-balanced-low-train-126-126.mat'));
              case 64
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-64-balanced-low-train-133-133.mat'));
              case 65
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-65-balanced-low-train-125-125.mat'));
              case 66
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-66-balanced-low-train-131-131.mat'));
              case 67
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-67-balanced-low-train-130-130.mat'));
              case 68
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-68-balanced-low-train-133-133.mat'));
              case 69
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-69-balanced-low-train-131-131.mat'));
              case 70
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-70-balanced-low-train-133-133.mat'));
              case 71
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-71-balanced-low-train-129-129.mat'));
              case 72
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-72-balanced-low-train-130-130.mat'));
              case 73
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-73-balanced-low-train-133-133.mat'));
              case 74
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-74-balanced-low-train-133-133.mat'));
              case 75
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-75-balanced-low-train-133-133.mat'));
              case 76
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-76-balanced-low-train-130-130.mat'));
              case 77
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-77-balanced-low-train-131-131.mat'));
              case 78
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-78-balanced-low-train-132-132.mat'));
              case 79
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-79-balanced-low-train-133-133.mat'));
              case 80
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-80-balanced-low-train-130-130.mat'));
              case 81
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-81-balanced-low-train-132-132.mat'));
              case 82
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-82-balanced-low-train-133-133.mat'));
              case 83
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-83-balanced-low-train-133-133.mat'));
              case 84
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-84-balanced-low-train-133-133.mat'));
              case 85
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-85-balanced-low-train-133-133.mat'));
              case 86
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-86-balanced-low-train-133-133.mat'));
              case 87
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-87-balanced-low-train-128-128.mat'));
              case 88
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-88-balanced-low-train-131-131.mat'));
              case 89
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-89-balanced-low-train-133-133.mat'));
              case 90
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-90-balanced-low-train-130-130.mat'));
              case 91
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-91-balanced-low-train-131-131.mat'));
              case 92
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-92-balanced-low-train-133-133.mat'));
              case 93
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-93-balanced-low-train-133-133.mat'));
              case 94
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-94-balanced-low-train-133-133.mat'));
              case 95
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-95-balanced-low-train-133-133.mat'));
              case 96
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-96-balanced-low-train-132-132.mat'));
              case 97
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-97-balanced-low-train-133-133.mat'));
              case 98
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-98-balanced-low-train-128-128.mat'));
              case 99
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-99-balanced-low-train-133-133.mat'));
              case 100
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-100-balanced-low-train-129-129.mat'));
              case 101
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-101-balanced-low-train-130-130.mat'));
              case 102
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-102-balanced-low-train-131-131.mat'));
              case 103
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-103-balanced-low-train-129-129.mat'));
              case 104
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-104-balanced-low-train-128-128.mat'));
            end
          case 'leave-one-out-unbalanced'
            switch fold_number
              case 01
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-01-unbalanced-train-133-12550.mat'));
              case 02
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-02-unbalanced-train-133-12563.mat'));
              case 03
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-03-unbalanced-train-130-12542.mat'));
              case 04
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-04-unbalanced-train-131-12570.mat'));
              case 05
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-05-unbalanced-train-133-12504.mat'));
              case 06
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-06-unbalanced-train-129-12509.mat'));
              case 07
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-07-unbalanced-train-133-12496.mat'));
              case 08
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-08-unbalanced-train-133-12538.mat'));
              case 09
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-09-unbalanced-train-133-12539.mat'));
              case 10
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-10-unbalanced-train-130-12540.mat'));
              case 11
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-11-unbalanced-train-130-12527.mat'));
              case 12
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-12-unbalanced-train-131-12541.mat'));
              case 13
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-13-unbalanced-train-133-12515.mat'));
              case 14
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-14-unbalanced-train-130-12550.mat'));
              case 15
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-15-unbalanced-train-133-12550.mat'));
              case 16
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-16-unbalanced-train-132-12505.mat'));
              case 17
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-17-unbalanced-train-133-12536.mat'));
              case 18
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-18-unbalanced-train-133-12468.mat'));
              case 19
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-19-unbalanced-train-133-12540.mat'));
              case 20
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-20-unbalanced-train-133-12493.mat'));
              case 21
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-21-unbalanced-train-133-12621.mat'));
              case 22
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-22-unbalanced-train-133-12547.mat'));
              case 23
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-23-unbalanced-train-133-12516.mat'));
              case 24
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-24-unbalanced-train-130-12523.mat'));
              case 25
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-25-unbalanced-train-126-12553.mat'));
              case 26
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-26-unbalanced-train-129-12532.mat'));
              case 27
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-27-unbalanced-train-131-12543.mat'));
              case 28
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-28-unbalanced-train-133-12503.mat'));
              case 29
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-29-unbalanced-train-131-12499.mat'));
              case 30
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-30-unbalanced-train-128-12558.mat'));
              case 31
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-31-unbalanced-train-133-12534.mat'));
              case 32
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-32-unbalanced-train-133-12522.mat'));
              case 33
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-33-unbalanced-train-133-12454.mat'));
              case 34
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-34-unbalanced-train-125-12553.mat'));
              case 35
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-35-unbalanced-train-133-12504.mat'));
              case 36
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-36-unbalanced-train-133-12515.mat'));
              case 37
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-37-unbalanced-train-128-12546.mat'));
              case 38
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-38-unbalanced-train-133-12501.mat'));
              case 39
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-39-unbalanced-train-131-12563.mat'));
              case 40
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-40-unbalanced-train-126-12554.mat'));
              case 41
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-41-unbalanced-train-133-12525.mat'));
              case 42
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-42-unbalanced-train-133-12498.mat'));
              case 43
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-43-unbalanced-train-132-12539.mat'));
              case 44
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-44-unbalanced-train-133-12547.mat'));
              case 45
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-45-unbalanced-train-133-12549.mat'));
              case 46
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-46-unbalanced-train-131-12551.mat'));
              case 47
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-47-unbalanced-train-128-12558.mat'));
              case 48
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-48-unbalanced-train-133-12507.mat'));
              case 49
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-49-unbalanced-train-133-12555.mat'));
              case 50
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-50-unbalanced-train-133-12520.mat'));
              case 51
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-51-unbalanced-train-129-12538.mat'));
              case 52
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-52-unbalanced-train-133-12512.mat'));
              case 53
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-53-unbalanced-train-133-12525.mat'));
              case 54
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-54-unbalanced-train-133-12559.mat'));
              case 55
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-55-unbalanced-train-133-12557.mat'));
              case 56
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-56-unbalanced-train-133-12545.mat'));
              case 57
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-57-unbalanced-train-130-12554.mat'));
              case 58
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-58-unbalanced-train-133-12506.mat'));
              case 59
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-59-unbalanced-train-128-12543.mat'));
              case 60
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-60-unbalanced-train-132-12544.mat'));
              case 61
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-61-unbalanced-train-131-12483.mat'));
              case 62
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-62-unbalanced-train-133-12556.mat'));
              case 63
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-63-unbalanced-train-130-12546.mat'));
              case 64
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-64-unbalanced-train-133-12535.mat'));
              case 65
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-65-unbalanced-train-133-12445.mat'));
              case 66
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-66-unbalanced-train-133-12513.mat'));
              case 67
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-67-unbalanced-train-133-12503.mat'));
              case 68
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-68-unbalanced-train-129-12548.mat'));
              case 69
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-69-unbalanced-train-133-12460.mat'));
              case 70
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-70-unbalanced-train-133-12535.mat'));
              case 71
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-71-unbalanced-train-133-12544.mat'));
              case 72
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-72-unbalanced-train-132-12566.mat'));
              case 73
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-73-unbalanced-train-133-12506.mat'));
              case 74
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-74-unbalanced-train-130-12517.mat'));
              case 75
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-75-unbalanced-train-131-12560.mat'));
              case 76
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-76-unbalanced-train-133-12539.mat'));
              case 77
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-77-unbalanced-train-133-12536.mat'));
              case 78
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-78-unbalanced-train-133-12506.mat'));
              case 79
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-79-unbalanced-train-131-12512.mat'));
              case 80
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-80-unbalanced-train-130-12483.mat'));
              case 81
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-81-unbalanced-train-131-12567.mat'));
              case 82
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-82-unbalanced-train-133-12526.mat'));
              case 83
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-83-unbalanced-train-129-12529.mat'));
              case 84
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-84-unbalanced-train-133-12527.mat'));
              case 85
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-85-unbalanced-train-125-12572.mat'));
              case 86
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-86-unbalanced-train-133-12595.mat'));
              case 87
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-87-unbalanced-train-133-12499.mat'));
              case 88
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-88-unbalanced-train-133-12551.mat'));
              case 89
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-89-unbalanced-train-133-12533.mat'));
              case 90
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-90-unbalanced-train-133-12545.mat'));
              case 91
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-91-unbalanced-train-133-12553.mat'));
              case 92
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-92-unbalanced-train-133-12524.mat'));
              case 93
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-93-unbalanced-train-126-12553.mat'));
              case 94
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-94-unbalanced-train-133-12498.mat'));
              case 95
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-95-unbalanced-train-133-12525.mat'));
              case 96
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-96-unbalanced-train-133-12507.mat'));
              case 97
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-97-unbalanced-train-133-12499.mat'));
              case 98
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-98-unbalanced-train-133-12531.mat'));
              case 99
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-99-unbalanced-train-130-12536.mat'));
              case 100
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-100-unbalanced-train-132-12560.mat'));
              case 101
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-101-unbalanced-train-133-12464.mat'));
              case 102
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-102-unbalanced-train-133-12524.mat'));
              case 103
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-103-unbalanced-train-132-12526.mat'));
              case 104
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-104-unbalanced-train-133-12467.mat'));
            end
          case 'leave-one-out-balanced-high'
            switch fold_number
              case 01
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-01-balanced-high-train-1064-1064.mat'));
              case 02
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-02-balanced-high-train-1064-1064.mat'));
              case 03
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-03-balanced-high-train-1064-1064.mat'));
              case 04
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-04-balanced-high-train-1064-1064.mat'));
              case 05
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-05-balanced-high-train-1024-1024.mat'));
              case 06
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-06-balanced-high-train-1064-1064.mat'));
              case 07
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-07-balanced-high-train-1064-1064.mat'));
              case 08
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-08-balanced-high-train-1040-1040.mat'));
              case 09
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-09-balanced-high-train-1064-1064.mat'));
              case 10
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-10-balanced-high-train-1048-1048.mat'));
              case 11
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-11-balanced-high-train-1064-1064.mat'));
              case 12
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-12-balanced-high-train-1040-1040.mat'));
              case 13
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-13-balanced-high-train-1064-1064.mat'));
              case 14
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-14-balanced-high-train-1064-1064.mat'));
              case 15
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-15-balanced-high-train-1064-1064.mat'));
              case 16
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-16-balanced-high-train-1024-1024.mat'));
              case 17
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-17-balanced-high-train-1064-1064.mat'));
              case 18
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-18-balanced-high-train-1048-1048.mat'));
              case 19
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-19-balanced-high-train-1064-1064.mat'));
              case 20
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-20-balanced-high-train-1064-1064.mat'));
              case 21
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-21-balanced-high-train-1048-1048.mat'));
              case 22
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-22-balanced-high-train-1064-1064.mat'));
              case 23
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-23-balanced-high-train-1064-1064.mat'));
              case 24
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-24-balanced-high-train-1064-1064.mat'));
              case 25
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-25-balanced-high-train-1056-1056.mat'));
              case 26
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-26-balanced-high-train-1064-1064.mat'));
              case 27
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-27-balanced-high-train-1064-1064.mat'));
              case 28
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-28-balanced-high-train-1048-1048.mat'));
              case 29
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-29-balanced-high-train-1048-1048.mat'));
              case 30
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-30-balanced-high-train-1064-1064.mat'));
              case 31
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-31-balanced-high-train-1056-1056.mat'));
              case 32
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-32-balanced-high-train-1064-1064.mat'));
              case 33
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-33-balanced-high-train-1048-1048.mat'));
              case 34
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-34-balanced-high-train-1040-1040.mat'));
              case 35
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-35-balanced-high-train-1056-1056.mat'));
              case 36
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-36-balanced-high-train-1064-1064.mat'));
              case 37
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-37-balanced-high-train-1032-1032.mat'));
              case 38
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-38-balanced-high-train-1064-1064.mat'));
              case 39
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-39-balanced-high-train-1064-1064.mat'));
              case 40
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-40-balanced-high-train-1064-1064.mat'));
              case 41
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-41-balanced-high-train-1048-1048.mat'));
              case 42
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-42-balanced-high-train-1064-1064.mat'));
              case 43
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-43-balanced-high-train-1032-1032.mat'));
              case 44
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-44-balanced-high-train-1000-1000.mat'));
              case 45
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-45-balanced-high-train-1064-1064.mat'));
              case 46
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-46-balanced-high-train-1064-1064.mat'));
              case 47
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-47-balanced-high-train-1048-1048.mat'));
              case 48
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-48-balanced-high-train-1056-1056.mat'));
              case 49
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-49-balanced-high-train-1064-1064.mat'));
              case 50
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-50-balanced-high-train-1024-1024.mat'));
              case 51
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-51-balanced-high-train-1064-1064.mat'));
              case 52
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-52-balanced-high-train-1056-1056.mat'));
              case 53
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-53-balanced-high-train-1064-1064.mat'));
              case 54
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-54-balanced-high-train-1064-1064.mat'));
              case 55
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-55-balanced-high-train-1064-1064.mat'));
              case 56
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-56-balanced-high-train-1064-1064.mat'));
              case 57
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-57-balanced-high-train-1040-1040.mat'));
              case 58
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-58-balanced-high-train-1048-1048.mat'));
              case 59
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-59-balanced-high-train-1064-1064.mat'));
              case 60
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-60-balanced-high-train-1064-1064.mat'));
              case 61
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-61-balanced-high-train-1064-1064.mat'));
              case 62
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-62-balanced-high-train-1064-1064.mat'));
              case 63
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-63-balanced-high-train-1064-1064.mat'));
              case 64
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-64-balanced-high-train-1032-1032.mat'));
              case 65
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-65-balanced-high-train-1064-1064.mat'));
              case 66
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-66-balanced-high-train-1056-1056.mat'));
              case 67
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-67-balanced-high-train-1064-1064.mat'));
              case 68
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-68-balanced-high-train-1032-1032.mat'));
              case 69
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-69-balanced-high-train-1040-1040.mat'));
              case 70
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-70-balanced-high-train-1064-1064.mat'));
              case 71
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-71-balanced-high-train-1040-1040.mat'));
              case 72
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-72-balanced-high-train-1064-1064.mat'));
              case 73
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-73-balanced-high-train-1064-1064.mat'));
              case 74
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-74-balanced-high-train-1064-1064.mat'));
              case 75
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-75-balanced-high-train-1064-1064.mat'));
              case 76
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-76-balanced-high-train-1024-1024.mat'));
              case 77
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-77-balanced-high-train-1064-1064.mat'));
              case 78
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-78-balanced-high-train-1008-1008.mat'));
              case 79
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-79-balanced-high-train-1040-1040.mat'));
              case 80
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-80-balanced-high-train-1064-1064.mat'));
              case 81
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-81-balanced-high-train-1040-1040.mat'));
              case 82
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-82-balanced-high-train-1064-1064.mat'));
              case 83
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-83-balanced-high-train-1064-1064.mat'));
              case 84
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-84-balanced-high-train-1048-1048.mat'));
              case 85
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-85-balanced-high-train-1064-1064.mat'));
              case 86
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-86-balanced-high-train-1064-1064.mat'));
              case 87
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-87-balanced-high-train-1064-1064.mat'));
              case 88
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-88-balanced-high-train-1008-1008.mat'));
              case 89
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-89-balanced-high-train-1032-1032.mat'));
              case 90
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-90-balanced-high-train-1064-1064.mat'));
              case 91
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-91-balanced-high-train-1064-1064.mat'));
              case 92
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-92-balanced-high-train-1064-1064.mat'));
              case 93
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-93-balanced-high-train-1000-1000.mat'));
              case 94
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-94-balanced-high-train-1064-1064.mat'));
              case 95
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-95-balanced-high-train-1064-1064.mat'));
              case 96
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-96-balanced-high-train-1040-1040.mat'));
              case 97
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-97-balanced-high-train-1008-1008.mat'));
              case 98
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-98-balanced-high-train-1040-1040.mat'));
              case 99
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-99-balanced-high-train-1064-1064.mat'));
              case 100
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-100-balanced-high-train-1064-1064.mat'));
              case 101
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-101-balanced-high-train-1064-1064.mat'));
              case 102
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-102-balanced-high-train-1064-1064.mat'));
              case 103
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-103-balanced-high-train-1064-1064.mat'));
              case 104
                tmp = load(fullfile(path_to_imdbs, dataset, posneg_balance, 'saved-two-class-prostate-v3-104-patients-pos2-neg1-patient-104-balanced-high-train-1064-1064.mat'));
            end
        end
      otherwise
        fprintf('TODO: implement!');
      end
  end
  imdb = tmp.imdb;
  afprintf(sprintf('[INFO] done!\n'));

  % print info
  printConsoleOutputSeparator();
  fh_imdb_utils = imdbTwoClassUtils;
  fh_imdb_utils.getImdbInfo(imdb, 1);
  printConsoleOutputSeparator();

