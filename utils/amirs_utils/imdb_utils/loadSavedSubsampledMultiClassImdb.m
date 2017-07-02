% -------------------------------------------------------------------------
function imdb = loadSavedSubsampledMultiClassImdb(dataset, posneg_balance)
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

  afprintf(sprintf('[INFO] Loading subsampled multi-class imdb (dataset: %s, posneg_balance: %s)\n', dataset, posneg_balance));
  path_to_imdbs = fullfile(getDevPath(), 'data', 'multi_class_subsampled_imdbs');
  switch dataset
    % case 'mnist-multi-class-subsampled'
    %   switch posneg_balance
    %     case 'balanced-38'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist', 'saved-multi-class-mnist-train-balance-38-test-balance-default.mat'));
    %     case 'balanced-100'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist', 'saved-multi-class-mnist-train-balance-100-test-balance-default.mat'));
    %     case 'balanced-266'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist', 'saved-multi-class-mnist-train-balance-266-test-balance-default.mat'));
    %     case 'balanced-707'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist', 'saved-multi-class-mnist-train-balance-707-test-balance-default.mat'));
    %     case 'balanced-1880'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist', 'saved-multi-class-mnist-train-balance-1880-test-balance-default.mat'));
    %     case 'balanced-5000'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist', 'saved-multi-class-mnist-train-balance-5000-test-balance-default.mat'));
    %   end
    % case 'cifar-multi-class-subsampled'
    %   switch posneg_balance
    %     case 'balanced-38'
    %       tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-multi-class-cifar-train-balance-38-test-balance-default.mat'));
    %     case 'balanced-100'
    %       tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-multi-class-cifar-train-balance-100-test-balance-default.mat'));
    %     case 'balanced-266'
    %       tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-multi-class-cifar-train-balance-266-test-balance-default.mat'));
    %     case 'balanced-707'
    %       tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-multi-class-cifar-train-balance-707-test-balance-default.mat'));
    %     case 'balanced-1880'
    %       tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-multi-class-cifar-train-balance-1880-test-balance-default.mat'));
    %     case 'balanced-5000'
    %       tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-multi-class-cifar-train-balance-5000-test-balance-default.mat'));
    %   end
    case 'mnist-multi-class-subsampled'
      % e.g.
      % posneg_balance = 'balanced-250'
      % file_name = 'saved-multi-class-mnist-train-balance-250-test-balance-default.mat'
      file_name = sprintf('saved-multi-class-mnist-train-balance-%s-test-balance-default.mat', posneg_balance(10:end));
      dataset_class = dataset(1:strfind(dataset, '-multi-class-subsampled') - 1);
      tmp = load(fullfile(path_to_imdbs, dataset_class, file_name));
    case 'mnist-784-multi-class-subsampled'
      % e.g.
      % posneg_balance = 'balanced-250'
      % file_name = 'saved-multi-class-mnist-train-balance-250-test-balance-default.mat'
      file_name = sprintf('saved-multi-784-class-mnist-train-balance-%s-test-balance-default.mat', posneg_balance(10:end));
      dataset_class = dataset(1:strfind(dataset, '-multi-class-subsampled') - 1);
      tmp = load(fullfile(path_to_imdbs, dataset_class, file_name));
    case 'cifar-multi-class-subsampled'
      % e.g.
      % posneg_balance = 'balanced-250'
      % file_name = 'saved-multi-class-cifar-train-balance-250-test-balance-default.mat'
      file_name = sprintf('saved-multi-class-cifar-train-balance-%s-test-balance-default.mat', posneg_balance(10:end));
      dataset_class = dataset(1:strfind(dataset, '-multi-class-subsampled') - 1);
      tmp = load(fullfile(path_to_imdbs, dataset_class, file_name));
    case 'svhn-multi-class-subsampled'
      switch posneg_balance
        case 'balanced-38'
          tmp = load(fullfile(path_to_imdbs, 'svhn', 'saved-multi-class-svhn-train-balance-38-test-balance-default.mat'));
        case 'balanced-100'
          tmp = load(fullfile(path_to_imdbs, 'svhn', 'saved-multi-class-svhn-train-balance-100-test-balance-default.mat'));
        case 'balanced-266'
          tmp = load(fullfile(path_to_imdbs, 'svhn', 'saved-multi-class-svhn-train-balance-266-test-balance-default.mat'));
        case 'balanced-707'
          tmp = load(fullfile(path_to_imdbs, 'svhn', 'saved-multi-class-svhn-train-balance-707-test-balance-default.mat'));
        case 'balanced-1880'
          tmp = load(fullfile(path_to_imdbs, 'svhn', 'saved-multi-class-svhn-train-balance-1880-test-balance-default.mat'));
        case 'balanced-5000'
          tmp = load(fullfile(path_to_imdbs, 'svhn', 'saved-multi-class-svhn-train-balance-4659-test-balance-default.mat'));
      end
    case 'cifar-no-white-multi-class-subsampled'
      switch posneg_balance
        case 'balanced-38'
          tmp = load(fullfile(path_to_imdbs, 'cifar-no-white', 'saved-multi-class-cifar-no-white-train-balance-38-test-balance-default.mat'));
        case 'balanced-100'
          tmp = load(fullfile(path_to_imdbs, 'cifar-no-white', 'saved-multi-class-cifar-no-white-train-balance-100-test-balance-default.mat'));
        case 'balanced-266'
          tmp = load(fullfile(path_to_imdbs, 'cifar-no-white', 'saved-multi-class-cifar-no-white-train-balance-266-test-balance-default.mat'));
        case 'balanced-707'
          tmp = load(fullfile(path_to_imdbs, 'cifar-no-white', 'saved-multi-class-cifar-no-white-train-balance-707-test-balance-default.mat'));
        case 'balanced-1880'
          tmp = load(fullfile(path_to_imdbs, 'cifar-no-white', 'saved-multi-class-cifar-no-white-train-balance-1880-test-balance-default.mat'));
        case 'balanced-5000'
          tmp = load(fullfile(path_to_imdbs, 'cifar-no-white', 'saved-multi-class-cifar-no-white-train-balance-5000-test-balance-default.mat'));
      end
    case 'stl-10-multi-class-subsampled'
      switch posneg_balance
        case 'balanced-38'
          tmp = load(fullfile(path_to_imdbs, 'stl-10', 'saved-multi-class-stl-10-train-balance-38-test-balance-default.mat'));
        case 'balanced-100'
          tmp = load(fullfile(path_to_imdbs, 'stl-10', 'saved-multi-class-stl-10-train-balance-100-test-balance-default.mat'));
        case 'balanced-266'
          tmp = load(fullfile(path_to_imdbs, 'stl-10', 'saved-multi-class-stl-10-train-balance-266-test-balance-default.mat'));
        case 'balanced-500'
          tmp = load(fullfile(path_to_imdbs, 'stl-10', 'saved-multi-class-stl-10-train-balance-500-test-balance-default.mat'));
      end
  end
  imdb = tmp.imdb;
  afprintf(sprintf('[INFO] done!\n'));




