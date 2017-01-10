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
  posneg_balance = getValueFromFieldOrDefault(input_opts, 'posneg_balance', 'balanced-low');
  fold_number = getValueFromFieldOrDefault(input_opts, 'fold_number', 1); % currently only implemented for prostate data

  afprintf(sprintf('[INFO] Loading imdb (dataset: %s, posneg_balance: %s)\n', dataset, posneg_balance));
  path_to_imdbs = fullfile(getDevPath(), 'data', 'two_class_imdbs');
  switch dataset
    case 'mnist-two-class-9-4'
      % currently fold number is not implemented.
      switch posneg_balance
        case 'balanced-low'
          tmp = load(fullfile(path_to_imdbs, 'mnist', 'saved-two-class-mnist-pos9-neg4-balanced-train-30-30.mat'));
        case 'unbalanced'
          tmp = load(fullfile(path_to_imdbs, 'mnist', 'saved-two-class-mnist-pos9-neg4-unbalanced-30-6000.mat'));
        case 'balanced-high'
          tmp = load(fullfile(path_to_imdbs, 'mnist', 'saved-two-class-mnist-pos9-neg4-balanced-train-6000-6000.mat'));
      end
    case 'cifar-two-class-deer-horse'
      % currently fold number is not implemented.
      switch posneg_balance
        case 'balanced-low'
          tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-two-class-cifar-pos5-neg8-balanced-train-25-25.mat'));
        case 'unbalanced'
          tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-two-class-cifar-pos5-neg8-unbalanced-25-5000.mat'));
        case 'balanced-high'
          tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-two-class-cifar-pos5-neg8-balanced-train-5000-5000.mat'));
        end
    case 'cifar-two-class-deer-truck'
      % currently fold number is not implemented.
      switch posneg_balance
        case 'balanced-low'
          tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-two-class-cifar-pos5-neg10-balanced-train-25-25.mat'));
        case 'unbalanced'
          tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-two-class-cifar-pos5-neg10-unbalanced-25-5000.mat'));
        case 'balanced-high'
          tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-two-class-cifar-pos5-neg10-balanced-train-5000-5000.mat'));
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
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'k=5-fold-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-unbalaned-train-51-655.mat'));
            case 2
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'k=5-fold-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-unbalaned-train-62-597.mat'));
            case 3
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'k=5-fold-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-unbalaned-train-68-544.mat'));
            case 4
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'k=5-fold-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-unbalaned-train-68-567.mat'));
            case 5
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'k=5-fold-unbalanced', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-unbalaned-train-71-493.mat'));
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
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'patient-1-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-77-659.mat'));
            case 2
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'patient-2-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-80-636.mat'));
            case 3
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'patient-3-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-80-692.mat'));
            case 4
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'patient-4-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-77-677.mat'));
            case 5
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'patient-5-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-71-703.mat'));
            case 6
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'patient-6-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-75-678.mat'));
            case 7
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'patient-7-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-70-692.mat'));
            case 8
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'patient-8-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-79-671.mat'));
            case 9
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'patient-9-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-74-688.mat'));
            case 10
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'patient-10-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-75-660.mat'));
            case 11
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'patient-11-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-77-677.mat'));
            case 12
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'patient-12-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-78-691.mat'));
            case 13
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'patient-13-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-76-691.mat'));
            case 14
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'patient-14-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-75-669.mat'));
            case 15
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'patient-15-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-78-603.mat'));
            case 16
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'patient-16-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-69-710.mat'));
            case 17
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'patient-17-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-75-701.mat'));
            case 18
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'patient-18-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-80-681.mat'));
            case 19
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'patient-19-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-77-696.mat'));
            case 20
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-unbalanced', 'patient-20-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-77-691.mat'));
          end
        case 'leave-one-out-balanced-high'
          switch fold_number
            case 1
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'patient-1-balanced-high-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-608-691.mat'));
            case 2
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'patient-2-balanced-high-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-600-701.mat'));
            case 3
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'patient-3-balanced-high-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-592-688.mat'));
            case 4
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'patient-4-balanced-high-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-640-636.mat'));
            case 5
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'patient-5-balanced-high-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-640-692.mat'));
            case 6
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'patient-6-balanced-high-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-616-691.mat'));
            case 7
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'patient-7-balanced-high-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-632-671.mat'));
            case 8
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'patient-8-balanced-high-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-600-669.mat'));
            case 9
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'patient-9-balanced-high-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-616-696.mat'));
            case 10
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'patient-10-balanced-high-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-616-659.mat'));
            case 11
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'patient-11-balanced-high-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-600-678.mat'));
            case 12
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'patient-12-balanced-high-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-624-603.mat'));
            case 13
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'patient-13-balanced-high-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-640-681.mat'));
            case 14
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'patient-14-balanced-high-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-552-710.mat'));
            case 15
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'patient-15-balanced-high-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-568-703.mat'));
            case 16
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'patient-16-balanced-high-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-560-692.mat'));
            case 17
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'patient-17-balanced-high-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-624-691.mat'));
            case 18
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'patient-18-balanced-high-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-616-677.mat'));
            case 19
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'patient-19-balanced-high-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-616-677.mat'));
            case 20
              tmp = load(fullfile(path_to_imdbs, 'prostate-v2-20-patients', 'leave-one-out-balanced-high', 'patient-20-balanced-high-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-600-660.mat'));
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

