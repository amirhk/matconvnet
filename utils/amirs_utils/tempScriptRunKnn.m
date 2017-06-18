% -------------------------------------------------------------------------
function tempScriptRunKNN()
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

  % -------------------------------------------------------------------------
  %                                                                 Get IMDBs
  % -------------------------------------------------------------------------
  % dataset = 'cifar';
  % posneg_balance = 'whatever';
  % dataset = 'cifar-multi-class-subsampled';
  % posneg_balance = 'balanced-266';
  % dataset = 'cifar-two-class-deer-truck';

  % dataset = 'gaussian-5D-160-train-40-test';
  % dataset = 'gaussian-10D-160-train-40-test';
  % dataset = 'gaussian-25D-160-train-40-test';
  % dataset = 'gaussian-50D-160-train-40-test';

  % dataset = 'gaussian-5D-400-train-100-test';
  % dataset = 'gaussian-10D-400-train-100-test';
  % dataset = 'gaussian-25D-400-train-100-test';
  % dataset = 'gaussian-50D-400-train-100-test';

  % dataset = 'gaussian-50D-800-train-200-test';
  % TODO: change name to include variance as well!!!!!!!!!!
  % generate more of the 0.1 variances and comapre

  dataset = 'gaussian-5D-160-train-40-test-0.1-var';
  dataset = 'gaussian-5D-160-train-40-test-1.0-var';
  dataset = 'gaussian-5D-160-train-40-test-10.0-var';

  dataset = 'gaussian-5D-400-train-100-test-0.1-var';
  dataset = 'gaussian-5D-400-train-100-test-1.0-var';
  dataset = 'gaussian-5D-400-train-100-test-10.0-var';

  dataset = 'gaussian-5D-800-train-200-test-0.1-var';
  dataset = 'gaussian-5D-800-train-200-test-1.0-var';
  dataset = 'gaussian-5D-800-train-200-test-10.0-var';

  posneg_balance = 'balanced-38';

  [~, experiments] = setupExperimentsUsingProjectedImbds(dataset, posneg_balance, 0);

  for i = 1 : numel(experiments)
    input_opts = {};
    input_opts.dataset = dataset;
    input_opts.imdb = experiments{i}.imdb;
    [~, experiments{i}.performance_summary] = testKnn(input_opts);
  end

  for i = 1 : numel(experiments)
    afprintf(sprintf( ...
      '[INFO] 1-KNN Results for `%s`: \t\t train acc = %.4f, test acc = %.4f \n\n', ...
      experiments{i}.title, ...
      experiments{i}.performance_summary.testing.train.accuracy, ...
      experiments{i}.performance_summary.testing.test.accuracy));
  end
