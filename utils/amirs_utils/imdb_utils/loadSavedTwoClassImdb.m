% -------------------------------------------------------------------------
function imdb = loadSavedTwoClassImdb(dataset, posneg_balance, fold_number, debug_flag)
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

  if debug_flag, afprintf(sprintf('[INFO] Loading two-class imdb (dataset: %s, posneg_balance: %s)\n', dataset, posneg_balance)); end;
  path_to_imdbs = fullfile(getDevPath(), 'data', 'two_class_imdbs');
  switch dataset
    % case 'mnist-784-two-class-9-4'
    %   % currently fold number is not implemented.
    %   switch posneg_balance
    %     case 'balanced-low'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'saved-two-class-mnist-pos9-neg4-balanced-low-train-29-29.mat'));
    %     case 'unbalanced'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'saved-two-class-mnist-pos9-neg4-unbalanced-train-29-6131.mat'));
    %     case 'balanced-high'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'saved-two-class-mnist-pos9-neg4-balanced-high-train-5851-6131.mat'));
    %     case 'balanced-38'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'balanced', 'saved-two-class-mnist-pos9-neg4-balanced-38-38-train-38-38.mat'));
    %     case 'balanced-100'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'balanced', 'saved-two-class-mnist-pos9-neg4-balanced-100-100-train-100-100.mat'));
    %     case 'balanced-266'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'balanced', 'saved-two-class-mnist-pos9-neg4-balanced-266-266-train-266-266.mat'));
    %     case 'balanced-707'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'balanced', 'saved-two-class-mnist-pos9-neg4-balanced-707-707-train-707-707.mat'));
    %     case 'balanced-1880'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'balanced', 'saved-two-class-mnist-pos9-neg4-balanced-1880-1880-train-1880-1880.mat'));
    %     case 'balanced-5000'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'balanced', 'saved-two-class-mnist-pos9-neg4-balanced-5000-5000-train-5000-5000.mat'));
    %   end

    case 'mnist-784-two-class-0-1'
      tmp = loadSpecialImdb(path_to_imdbs, dataset, posneg_balance);
    case 'mnist-784-two-class-0-2'
      tmp = loadSpecialImdb(path_to_imdbs, dataset, posneg_balance);
    case 'mnist-784-two-class-0-3'
      tmp = loadSpecialImdb(path_to_imdbs, dataset, posneg_balance);
    case 'mnist-784-two-class-0-4'
      tmp = loadSpecialImdb(path_to_imdbs, dataset, posneg_balance);
    case 'mnist-784-two-class-5-0'
      tmp = loadSpecialImdb(path_to_imdbs, dataset, posneg_balance);
    case 'mnist-784-two-class-7-2'
      tmp = loadSpecialImdb(path_to_imdbs, dataset, posneg_balance);
    case 'mnist-784-two-class-8-2'
      tmp = loadSpecialImdb(path_to_imdbs, dataset, posneg_balance);
    case 'mnist-784-two-class-8-3'
      tmp = loadSpecialImdb(path_to_imdbs, dataset, posneg_balance);
    case 'mnist-784-two-class-4-9'
      tmp = loadSpecialImdb(path_to_imdbs, dataset, posneg_balance);
    case 'mnist-784-two-class-6-9'
      tmp = loadSpecialImdb(path_to_imdbs, dataset, posneg_balance);




    case 'svhn-two-class-1-0'
      tmp = loadSpecialImdb(path_to_imdbs, dataset, posneg_balance);
    case 'svhn-two-class-2-0'
      tmp = loadSpecialImdb(path_to_imdbs, dataset, posneg_balance);
    case 'svhn-two-class-3-0'
      tmp = loadSpecialImdb(path_to_imdbs, dataset, posneg_balance);
    case 'svhn-two-class-4-0'
      tmp = loadSpecialImdb(path_to_imdbs, dataset, posneg_balance);
    case 'svhn-two-class-5-0'
      tmp = loadSpecialImdb(path_to_imdbs, dataset, posneg_balance);
    case 'svhn-two-class-7-2'
      tmp = loadSpecialImdb(path_to_imdbs, dataset, posneg_balance);
    case 'svhn-two-class-8-2'
      tmp = loadSpecialImdb(path_to_imdbs, dataset, posneg_balance);
    case 'svhn-two-class-8-3'
      tmp = loadSpecialImdb(path_to_imdbs, dataset, posneg_balance);
    case 'svhn-two-class-9-4'
      tmp = loadSpecialImdb(path_to_imdbs, dataset, posneg_balance);
    case 'svhn-two-class-9-6'
      tmp = loadSpecialImdb(path_to_imdbs, dataset, posneg_balance);

    %   switch posneg_balance
    %     case 'balanced-10'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'balanced', 'saved-two-class-mnist-784-pos0-neg1-balanced-10-10-train-10-10.mat'));
    %     case 'balanced-50'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'balanced', 'saved-two-class-mnist-784-pos0-neg1-balanced-50-50-train-50-50.mat'));
    %     case 'balanced-100'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'balanced', 'saved-two-class-mnist-784-pos0-neg1-balanced-100-100-train-100-100.mat'));
    %     case 'balanced-250'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'balanced', 'saved-two-class-mnist-784-pos0-neg1-balanced-250-250-train-250-250.mat'));
    %     case 'balanced-500'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'balanced', 'saved-two-class-mnist-784-pos0-neg1-balanced-500-500-train-500-500.mat'));
    %     case 'balanced-1000'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'balanced', 'saved-two-class-mnist-784-pos0-neg1-balanced-1000-1000-train-1000-1000.mat'));
    %     case 'balanced-2500'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'balanced', 'saved-two-class-mnist-784-pos0-neg1-balanced-2500-2500-train-2500-2500.mat'));
    %   end
    % case 'mnist-784-two-class-0-1'
    %   switch posneg_balance
    %     case 'balanced-10'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'balanced', 'saved-two-class-mnist-784-pos8-neg3-balanced-10-10-train-10-10.mat'));
    %     case 'balanced-50'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'balanced', 'saved-two-class-mnist-784-pos8-neg3-balanced-50-50-train-50-50.mat'));
    %     case 'balanced-100'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'balanced', 'saved-two-class-mnist-784-pos8-neg3-balanced-100-100-train-100-100.mat'));
    %     case 'balanced-250'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'balanced', 'saved-two-class-mnist-784-pos8-neg3-balanced-250-250-train-250-250.mat'));
    %     case 'balanced-500'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'balanced', 'saved-two-class-mnist-784-pos8-neg3-balanced-500-500-train-500-500.mat'));
    %     case 'balanced-1000'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'balanced', 'saved-two-class-mnist-784-pos8-neg3-balanced-1000-1000-train-1000-1000.mat'));
    %     case 'balanced-2500'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'balanced', 'saved-two-class-mnist-784-pos8-neg3-balanced-2500-2500-train-2500-2500.mat'));
    %   end

    % case 'mnist-784-two-class-0-1'
    %   switch posneg_balance
    %     case 'balanced-5000'
    %       tmp = load(fullfile(path_to_imdbs, 'mnist-784', 'balanced', 'saved-two-class-mnist-pos1-neg2-balanced-5000-5000-train-5000-5000.mat'));
    %   end



























    case 'imagenet-tiny-two-class-school-bus-remote-control'
      switch posneg_balance
        case 'balanced-10'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg24-balanced-10-10-train-10-10.mat'));
        case 'balanced-50'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg24-balanced-50-50-train-50-50.mat'));
        case 'balanced-100'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg24-balanced-100-100-train-100-100.mat'));
        case 'balanced-250'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg24-balanced-250-250-train-250-250.mat'));
        case 'balanced-500'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg24-balanced-500-500-train-500-500.mat'));
      end

    case 'imagenet-tiny-two-class-school-bus-rocking-chair'
      switch posneg_balance
        case 'balanced-10'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg34-balanced-10-10-train-10-10.mat'));
        case 'balanced-50'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg34-balanced-50-50-train-50-50.mat'));
        case 'balanced-100'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg34-balanced-100-100-train-100-100.mat'));
        case 'balanced-250'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg34-balanced-250-250-train-250-250.mat'));
        case 'balanced-500'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg34-balanced-500-500-train-500-500.mat'));
      end

    case 'imagenet-tiny-two-class-school-bus-monarch-butterfly'
      switch posneg_balance
        case 'balanced-10'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg36-balanced-10-10-train-10-10.mat'));
        case 'balanced-50'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg36-balanced-50-50-train-50-50.mat'));
        case 'balanced-100'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg36-balanced-100-100-train-100-100.mat'));
        case 'balanced-250'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg36-balanced-250-250-train-250-250.mat'));
        case 'balanced-500'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg36-balanced-500-500-train-500-500.mat'));
      end

    case 'imagenet-tiny-two-class-school-bus-steel-arch-bridge'
      switch posneg_balance
        case 'balanced-10'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg109-balanced-10-10-train-10-10.mat'));
        case 'balanced-50'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg109-balanced-50-50-train-50-50.mat'));
        case 'balanced-100'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg109-balanced-100-100-train-100-100.mat'));
        case 'balanced-250'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg109-balanced-250-250-train-250-250.mat'));
        case 'balanced-500'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg109-balanced-500-500-train-500-500.mat'));
      end

    case 'imagenet-tiny-two-class-remote-control-rocking-chair'
      switch posneg_balance
        case 'balanced-10'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos24-neg34-balanced-10-10-train-10-10.mat'));
        case 'balanced-50'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos24-neg34-balanced-50-50-train-50-50.mat'));
        case 'balanced-100'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos24-neg34-balanced-100-100-train-100-100.mat'));
        case 'balanced-250'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos24-neg34-balanced-250-250-train-250-250.mat'));
        case 'balanced-500'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos24-neg34-balanced-500-500-train-500-500.mat'));
      end

    case 'imagenet-tiny-two-class-monarch-butterfly-lion'
      switch posneg_balance
        case 'balanced-10'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos36-neg44-balanced-10-10-train-10-10.mat'));
        case 'balanced-50'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos36-neg44-balanced-50-50-train-50-50.mat'));
        case 'balanced-100'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos36-neg44-balanced-100-100-train-100-100.mat'));
        case 'balanced-250'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos36-neg44-balanced-250-250-train-250-250.mat'));
        case 'balanced-500'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos36-neg44-balanced-500-500-train-500-500.mat'));
      end

    case 'imagenet-tiny-two-class-monarch-butterfly-steel-arch-bridge'
      switch posneg_balance
        case 'balanced-10'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos36-neg109-balanced-10-10-train-10-10.mat'));
        case 'balanced-50'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos36-neg109-balanced-50-50-train-50-50.mat'));
        case 'balanced-100'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos36-neg109-balanced-100-100-train-100-100.mat'));
        case 'balanced-250'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos36-neg109-balanced-250-250-train-250-250.mat'));
        case 'balanced-500'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos36-neg109-balanced-500-500-train-500-500.mat'));
      end

    case 'imagenet-tiny-two-class-lion-brown-bear'
      switch posneg_balance
        case 'balanced-10'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos44-neg73-balanced-10-10-train-10-10.mat'));
        case 'balanced-50'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos44-neg73-balanced-50-50-train-50-50.mat'));
        case 'balanced-100'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos44-neg73-balanced-100-100-train-100-100.mat'));
        case 'balanced-250'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos44-neg73-balanced-250-250-train-250-250.mat'));
        case 'balanced-500'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos44-neg73-balanced-500-500-train-500-500.mat'));
      end

    case 'imagenet-tiny-two-class-lion-german-shepherd'
      switch posneg_balance
        case 'balanced-10'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos44-neg192-balanced-10-10-train-10-10.mat'));
        case 'balanced-50'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos44-neg192-balanced-50-50-train-50-50.mat'));
        case 'balanced-100'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos44-neg192-balanced-100-100-train-100-100.mat'));
        case 'balanced-250'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos44-neg192-balanced-250-250-train-250-250.mat'));
        case 'balanced-500'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos44-neg192-balanced-500-500-train-500-500.mat'));
      end

    case 'imagenet-tiny-two-class-brown-bear-german-shepherd'
      switch posneg_balance
        case 'balanced-10'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos73-neg192-balanced-10-10-train-10-10.mat'));
        case 'balanced-50'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos73-neg192-balanced-50-50-train-50-50.mat'));
        case 'balanced-100'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos73-neg192-balanced-100-100-train-100-100.mat'));
        case 'balanced-250'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos73-neg192-balanced-250-250-train-250-250.mat'));
        case 'balanced-500'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos73-neg192-balanced-500-500-train-500-500.mat'));
      end

    case 'imagenet-tiny-two-class-school-bus-german-shepherd'
      switch posneg_balance
        case 'balanced-10'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg192-balanced-10-10-train-10-10.mat'));
        case 'balanced-50'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg192-balanced-50-50-train-50-50.mat'));
        case 'balanced-100'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg192-balanced-100-100-train-100-100.mat'));
        case 'balanced-250'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg192-balanced-250-250-train-250-250.mat'));
        case 'balanced-500'
          tmp = load(fullfile(path_to_imdbs, 'imagenet-tiny', 'balanced', 'saved-two-class-imagenet-tiny-pos8-neg192-balanced-500-500-train-500-500.mat'));
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
    case 'uci-ion'
      tmp = load(fullfile(path_to_imdbs, 'uci-ion', 'uci-ion.mat'));
    case 'uci-spam'
      tmp = load(fullfile(path_to_imdbs, 'uci-spam', 'uci-spam.mat'));
    otherwise
      fprintf('TODO: implement!');
  end
  imdb = tmp.imdb;
  if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;



% -------------------------------------------------------------------------
function tmp = loadSpecialImdb(path_to_imdbs, dataset, posneg_balance)
% -------------------------------------------------------------------------
  dataset_class = dataset(1:strfind(dataset, '-two-class') - 1);
  positive_class_number = str2num(getStringParameterStartingAtIndex(dataset, length(dataset_class) + 12)); % -two-class-
  negative_class_number = str2num(getStringParameterStartingAtIndex(dataset, length(dataset_class) + 12 + length(num2str(positive_class_number)) + 1));
  balance_number = posneg_balance(10:end);

  file_name = sprintf('saved-two-class-%s-pos%d-neg%d-balanced-%s-%s-train-%s-%s.mat', dataset_class, positive_class_number, negative_class_number, balance_number, balance_number, balance_number, balance_number);
  tmp = load(fullfile(path_to_imdbs, dataset_class, 'balanced', file_name));


