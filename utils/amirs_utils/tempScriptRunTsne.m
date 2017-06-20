% -------------------------------------------------------------------------
function tempScriptRunTsne(dataset, posneg_balance, save_results)
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
  %                                                                     Setup
  % -------------------------------------------------------------------------
  [~, experiments] = setupExperimentsUsingProjectedImbds(dataset, posneg_balance, 1);

  for i = 1 : numel(experiments)
    imdb = experiments{i}.imdb;
    number_of_features = size(imdb.images.data, 1) * size(imdb.images.data, 2) * size(imdb.images.data, 3);

    % Set parameters
    no_dims = 2;
    initial_dims = 50;
    % perplexity = 30;
    if initial_dims >= number_of_features
      initial_dims = number_of_features;
    end

    vectorized_data = reshape(imdb.images.data, number_of_features, [])';
    labels = imdb.images.labels;
    is_train = imdb.images.set == 1;
    is_test = imdb.images.set == 3;

    vectorized_data_train = vectorized_data(is_train, :);
    vectorized_data_test = vectorized_data(is_test, :);
    % vectorized_data_train = vectorized_data_train';
    % vectorized_data_test = vectorized_data_test';
    labels_train = labels(is_train);
    labels_test = labels(is_test);

    h = figure,
    j =  1;
    perplexity_array = [2, 5, 30, 50, 100];
    % perplexity_array = [30];
    for perplexity = perplexity_array
      subplot(1, numel(perplexity_array), j);
      hold on
      title(sprintf('perplexity: %d', perplexity));
      % Run t−SNE
      mappedX = tsne(vectorized_data_train, [], no_dims, initial_dims, perplexity);
      % Plot results
      % figure,
      % title(experiments{i}.title),
      gscatter(mappedX(:,1), mappedX(:,2), labels_train);
      hold off
      j = j + 1;
    end

    tmp_string = sprintf('t-SNE - %s - %s', experiments{i}.imdb.name, experiments{i}.title);
    suptitle(tmp_string);
    if save_results
      saveas(h, fullfile(getDevPath(), 'temp_images', sprintf('%s.png', tmp_string)));
    end
  end

