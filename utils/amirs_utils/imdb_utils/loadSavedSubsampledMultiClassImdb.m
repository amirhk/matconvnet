% -------------------------------------------------------------------------
function imdb = loadSavedSubsampledMultiClassImdb(dataset, posneg_balance, debug_flag)
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

  if debug_flag, afprintf(sprintf('[INFO] Loading subsampled multi-class imdb (dataset: %s, posneg_balance: %s)\n', dataset, posneg_balance)); end;
  path_to_imdbs = fullfile(getDevPath(), 'data', 'multi_class_subsampled_imdbs');


  % e.g.
  % posneg_balance = 'balanced-250'
  % file_name = 'saved-multi-class-mnist-train-balance-250-test-balance-default.mat'
  balance_number = posneg_balance(10:end);
  dataset_class = dataset(1:strfind(dataset, '-multi-class-subsampled') - 1);
  if strcmp(dataset_class, 'pathology')
    file_name = sprintf('saved-multi-class-%s-train-balance-%s-test-balance-1000.mat', dataset_class, balance_number);
  else
    % file_name = sprintf('saved-multi-class-%s-train-balance-%s-test-balance-default.mat', dataset_class, balance_number);
    file_name = sprintf('saved-multi-class-%s-train-balance-25-test-balance-25.mat', dataset_class);
  end
  tmp = load(fullfile(path_to_imdbs, dataset_class, file_name));

  imdb = tmp.imdb;
  if debug_flag, afprintf(sprintf('[INFO] done!\n')); end;




