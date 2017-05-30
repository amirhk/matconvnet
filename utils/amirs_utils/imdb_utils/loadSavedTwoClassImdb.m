% -------------------------------------------------------------------------
function imdb = loadSavedTwoClassImdb(dataset, posneg_balance, fold_number)
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
        case 'balanced-38'
          tmp = load(fullfile(path_to_imdbs, 'mnist', 'balanced', 'saved-two-class-mnist-pos9-neg4-balanced-38-38-train-38-38.mat'));
        case 'balanced-100'
          tmp = load(fullfile(path_to_imdbs, 'mnist', 'balanced', 'saved-two-class-mnist-pos9-neg4-balanced-100-100-train-100-100.mat'));
        case 'balanced-266'
          tmp = load(fullfile(path_to_imdbs, 'mnist', 'balanced', 'saved-two-class-mnist-pos9-neg4-balanced-266-266-train-266-266.mat'));
        case 'balanced-707'
          tmp = load(fullfile(path_to_imdbs, 'mnist', 'balanced', 'saved-two-class-mnist-pos9-neg4-balanced-707-707-train-707-707.mat'));
        case 'balanced-1880'
          tmp = load(fullfile(path_to_imdbs, 'mnist', 'balanced', 'saved-two-class-mnist-pos9-neg4-balanced-1880-1880-train-1880-1880.mat'));
        case 'balanced-5000'
          tmp = load(fullfile(path_to_imdbs, 'mnist', 'balanced', 'saved-two-class-mnist-pos9-neg4-balanced-5000-5000-train-5000-5000.mat'));
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
        case 'balanced-38'
          tmp = load(fullfile(path_to_imdbs, 'cifar', 'balanced', 'saved-two-class-cifar-pos5-neg10-balanced-38-38-train-38-38.mat'));
        case 'balanced-100'
          tmp = load(fullfile(path_to_imdbs, 'cifar', 'balanced', 'saved-two-class-cifar-pos5-neg10-balanced-100-100-train-100-100.mat'));
        case 'balanced-266'
          tmp = load(fullfile(path_to_imdbs, 'cifar', 'balanced', 'saved-two-class-cifar-pos5-neg10-balanced-266-266-train-266-266.mat'));
        case 'balanced-707'
          tmp = load(fullfile(path_to_imdbs, 'cifar', 'balanced', 'saved-two-class-cifar-pos5-neg10-balanced-707-707-train-707-707.mat'));
        case 'balanced-1880'
          tmp = load(fullfile(path_to_imdbs, 'cifar', 'balanced', 'saved-two-class-cifar-pos5-neg10-balanced-1880-1880-train-1880-1880.mat'));
        case 'balanced-5000'
          tmp = load(fullfile(path_to_imdbs, 'cifar', 'balanced', 'saved-two-class-cifar-pos5-neg10-balanced-5000-5000-train-5000-5000.mat'));
      end
    case 'cifar-no-white-two-class-deer-truck'
      % currently fold number is not implemented.
      switch posneg_balance
        case 'balanced-low'
          tmp = load(fullfile(path_to_imdbs, 'cifar-no-white', 'saved-two-class-cifar-pos5-neg10-balanced-low-train-25-25.mat'));
        case 'unbalanced'
          tmp = load(fullfile(path_to_imdbs, 'cifar-no-white', 'saved-two-class-cifar-pos5-neg10-unbalanced-train-25-5000.mat'));
        case 'balanced-high'
          tmp = load(fullfile(path_to_imdbs, 'cifar-no-white', 'saved-two-class-cifar-pos5-neg10-balanced-high-train-5000-5000.mat'));
        case 'balanced-38'
          tmp = load(fullfile(path_to_imdbs, 'cifar-no-white', 'balanced', 'saved-two-class-cifar-pos5-neg10-balanced-38-38-train-38-38.mat'));
        case 'balanced-100'
          tmp = load(fullfile(path_to_imdbs, 'cifar-no-white', 'balanced', 'saved-two-class-cifar-pos5-neg10-balanced-100-100-train-100-100.mat'));
        case 'balanced-266'
          tmp = load(fullfile(path_to_imdbs, 'cifar-no-white', 'balanced', 'saved-two-class-cifar-pos5-neg10-balanced-266-266-train-266-266.mat'));
        case 'balanced-707'
          tmp = load(fullfile(path_to_imdbs, 'cifar-no-white', 'balanced', 'saved-two-class-cifar-pos5-neg10-balanced-707-707-train-707-707.mat'));
        case 'balanced-1880'
          tmp = load(fullfile(path_to_imdbs, 'cifar-no-white', 'balanced', 'saved-two-class-cifar-pos5-neg10-balanced-1880-1880-train-1880-1880.mat'));
        case 'balanced-5000'
          tmp = load(fullfile(path_to_imdbs, 'cifar-no-white', 'balanced', 'saved-two-class-cifar-pos5-neg10-balanced-5000-5000-train-5000-5000.mat'));
      end
    case 'stl-10-two-class-airplane-bird'
      % currently fold number is not implemented.
      switch posneg_balance
        case 'balanced-low'
          tmp = load(fullfile(path_to_imdbs, 'stl-10', 'saved-two-class-stl-10-pos1-neg2-balanced-low-train-25-25.mat'));
        case 'unbalanced'
          tmp = load(fullfile(path_to_imdbs, 'stl-10', 'saved-two-class-stl-10-pos1-neg2-unbalanced-train-25-500.mat'));
        case 'balanced-high'
          tmp = load(fullfile(path_to_imdbs, 'stl-10', 'saved-two-class-stl-10-pos1-neg2-balanced-high-train-500-500.mat'));
        case 'balanced-38'
          tmp = load(fullfile(path_to_imdbs, 'stl-10', 'balanced', 'saved-two-class-stl-10-pos1-neg2-balanced-38-38-train-38-38.mat'));
        case 'balanced-100'
          tmp = load(fullfile(path_to_imdbs, 'stl-10', 'balanced', 'saved-two-class-stl-10-pos1-neg2-balanced-100-100-train-100-100.mat'));
        case 'balanced-266'
          tmp = load(fullfile(path_to_imdbs, 'stl-10', 'balanced', 'saved-two-class-stl-10-pos1-neg2-balanced-266-266-train-266-266.mat'));
        case 'balanced-500'
          tmp = load(fullfile(path_to_imdbs, 'stl-10', 'balanced', 'saved-two-class-stl-10-pos1-neg2-balanced-500-500-train-500-500.mat'));
      end
    case 'stl-10-two-class-airplane-cat'
      % currently fold number is not implemented.
      switch posneg_balance
        case 'balanced-38'
          tmp = load(fullfile(path_to_imdbs, 'stl-10', 'balanced', 'saved-two-class-stl-10-pos1-neg4-balanced-38-38-train-38-38.mat'));
        case 'balanced-100'
          tmp = load(fullfile(path_to_imdbs, 'stl-10', 'balanced', 'saved-two-class-stl-10-pos1-neg4-balanced-100-100-train-100-100.mat'));
        case 'balanced-266'
          tmp = load(fullfile(path_to_imdbs, 'stl-10', 'balanced', 'saved-two-class-stl-10-pos1-neg4-balanced-266-266-train-266-266.mat'));
        case 'balanced-500'
          tmp = load(fullfile(path_to_imdbs, 'stl-10', 'balanced', 'saved-two-class-stl-10-pos1-neg4-balanced-500-500-train-500-500.mat'));
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
        case 'balanced-38'
          tmp = load(fullfile(path_to_imdbs, 'svhn', 'balanced', 'saved-two-class-svhn-pos9-neg4-balanced-38-38-train-38-38.mat'));
        case 'balanced-100'
          tmp = load(fullfile(path_to_imdbs, 'svhn', 'balanced', 'saved-two-class-svhn-pos9-neg4-balanced-100-100-train-100-100.mat'));
        case 'balanced-266'
          tmp = load(fullfile(path_to_imdbs, 'svhn', 'balanced', 'saved-two-class-svhn-pos9-neg4-balanced-266-266-train-266-266.mat'));
        case 'balanced-707'
          tmp = load(fullfile(path_to_imdbs, 'svhn', 'balanced', 'saved-two-class-svhn-pos9-neg4-balanced-707-707-train-707-707.mat'));
        case 'balanced-1880'
          tmp = load(fullfile(path_to_imdbs, 'svhn', 'balanced', 'saved-two-class-svhn-pos9-neg4-balanced-1880-1880-train-1880-1880.mat'));
        case 'balanced-5000'
          tmp = load(fullfile(path_to_imdbs, 'svhn', 'balanced', 'saved-two-class-svhn-pos9-neg4-balanced-4659-4659-train-4659-4659.mat'));
      end
    case 'prostate-v2-20-patients'
      tmp = loadSavedProstateImdb(dataset, posneg_balance, fold_number)
    case 'prostate-v3-104-patients'
      tmp = loadSavedProstateImdb(dataset, posneg_balance, fold_number)
    otherwise
      fprintf('TODO: implement!');
  end
  imdb = tmp.imdb;
  afprintf(sprintf('[INFO] done!\n'));






