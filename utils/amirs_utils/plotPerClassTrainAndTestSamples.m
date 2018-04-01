% -------------------------------------------------------------------------
function plotPerClassTrainAndTestSamples(X_train, Y_train, X_test, Y_test)
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

  assert(numel(unique(Y_train)) == numel(unique(Y_test)));

  indices = {};
  for i = 1:numel(unique(Y_train))
    indices.train.(sprintf('class_%d',i)) = Y_train == i;
    indices.test.(sprintf('class_%d',i)) = Y_test == i;
  end

  data_train_per_class = {};
  data_test_per_class = {};
  for i = 1:numel(unique(Y_train))
    data_train_per_class.(sprintf('class_%d',i)) = X_train(:, indices.train.(sprintf('class_%d',i)));
    data_test_per_class.(sprintf('class_%d',i)) = X_test(:, indices.test.(sprintf('class_%d',i)));
  end

  cmap = colormap(parula(10));
  if numel(unique(Y_train)) == 2
    cmap = [cmap(1,:); cmap(10,:)];
  elseif numel(unique(Y_train)) == 3
    cmap = [cmap(1,:); cmap(4,:); cmap(10,:)];
  end

  hold on,
  for i = 1:numel(unique(Y_train))
    scatter( ...
      data_train_per_class.(sprintf('class_%d',i))(1,:), ...
      data_train_per_class.(sprintf('class_%d',i))(2,:), ...
      'filled', 'MarkerFaceColor', cmap(i,:));
    scatter( ...
      data_train_per_class.(sprintf('class_%d',i))(1,:), ...
      data_train_per_class.(sprintf('class_%d',i))(2,:), ...
      'MarkerFaceColor', cmap(i,:));
  end
  hold off,
