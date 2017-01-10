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

  if strcmp(dataset, 'cifar') || ...
    strcmp(dataset, 'coil-100') || ...
    strcmp(dataset, 'mnist') || ...
    strcmp(dataset, 'stl-10') || ...
    strcmp(dataset, 'svhn')
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
            tmp = load(fullfile(path_to_imdbs, 'mnist', 'saved-two-class-mnist-pos9-neg4-balanced-low-train-30-30.mat'));
          case 'unbalanced'
            tmp = load(fullfile(path_to_imdbs, 'mnist', 'saved-two-class-mnist-pos9-neg4-unbalanced-train-30-6000.mat'));
          case 'balanced-high'
            tmp = load(fullfile(path_to_imdbs, 'mnist', 'saved-two-class-mnist-pos9-neg4-balanced-high-train-6000-6000.mat'));
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
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'k=5-fold-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-unbalanced-train-51-655.mat'));
              case 2
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'k=5-fold-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-unbalanced-train-62-597.mat'));
              case 3
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'k=5-fold-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-unbalanced-train-68-544.mat'));
              case 4
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'k=5-fold-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-unbalanced-train-68-567.mat'));
              case 5
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'k=5-fold-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-unbalanced-train-71-493.mat'));
            end
          case 'k=5-fold-balanced-high'
            switch fold_number
              case 1
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'k=5-fold-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-balanced-high-train-488-597.mat'));
              case 2
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'k=5-fold-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-balanced-high-train-504-498.mat'));
              case 3
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'k=5-fold-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-balanced-high-train-512-574.mat'));
              case 4
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'k=5-fold-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-balanced-high-train-512-599.mat'));
              case 5
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'k=5-fold-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-balanced-high-train-544-588.mat'));
            end
          case 'leave-one-out-unbalanced'
            switch fold_number
              case 1
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-01-unbalanced-train-77-677.mat'));
              case 2
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-02-unbalanced-train-77-659.mat'));
              case 3
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-03-unbalanced-train-75-669.mat'));
              case 4
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-04-unbalanced-train-76-691.mat'));
              case 5
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-05-unbalanced-train-70-692.mat'));
              case 6
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-06-unbalanced-train-80-636.mat'));
              case 7
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-07-unbalanced-train-80-692.mat'));
              case 8
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-08-unbalanced-train-78-691.mat'));
              case 9
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-09-unbalanced-train-75-678.mat'));
              case 10
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-10-unbalanced-train-74-688.mat'));
              case 11
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-11-unbalanced-train-75-701.mat'));
              case 12
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-12-unbalanced-train-77-696.mat'));
              case 13
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-13-unbalanced-train-71-703.mat'));
              case 14
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-14-unbalanced-train-78-603.mat'));
              case 15
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-15-unbalanced-train-79-671.mat'));
              case 16
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-16-unbalanced-train-77-677.mat'));
              case 17
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-17-unbalanced-train-77-691.mat'));
              case 18
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-18-unbalanced-train-75-660.mat'));
              case 19
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-19-unbalanced-train-80-681.mat'));
              case 20
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-20-unbalanced-train-69-710.mat'));
            end
          case 'leave-one-out-balanced-high'
            switch fold_number
              case 1
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-01-balanced-high-train-568-703.mat'));
              case 2
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-02-balanced-high-train-600-669.mat'));
              case 3
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-03-balanced-high-train-592-688.mat'));
              case 4
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-04-balanced-high-train-608-691.mat'));
              case 5
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-05-balanced-high-train-616-659.mat'));
              case 6
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-06-balanced-high-train-560-692.mat'));
              case 7
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-07-balanced-high-train-624-691.mat'));
              case 8
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-08-balanced-high-train-640-636.mat'));
              case 9
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-09-balanced-high-train-600-660.mat'));
              case 10
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-10-balanced-high-train-624-603.mat'));
              case 11
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-11-balanced-high-train-600-701.mat'));
              case 12
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-12-balanced-high-train-640-681.mat'));
              case 13
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-13-balanced-high-train-616-677.mat'));
              case 14
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-14-balanced-high-train-552-710.mat'));
              case 15
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-15-balanced-high-train-616-696.mat'));
              case 16
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-16-balanced-high-train-616-691.mat'));
              case 17
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-17-balanced-high-train-640-692.mat'));
              case 18
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-18-balanced-high-train-600-678.mat'));
              case 19
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-19-balanced-high-train-632-671.mat'));
              case 20
                tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-patient-20-balanced-high-train-616-677.mat'));
            end
          otherwise
            fprintf('TODO: implement!');
        end
    end
    imdb = tmp.imdb;
  end
  afprintf(sprintf('[INFO] done!\n'));

  % print info
  printConsoleOutputSeparator();
  fh_imdb_utils = imdbTwoClassUtils;
  fh_imdb_utils.getImdbInfo(imdb, 1);
  printConsoleOutputSeparator();

