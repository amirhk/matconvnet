% -------------------------------------------------------------------------
function dumb_array = tmpScriptCalculateDistances2(dataset, posneg_balance, ignore_todo_remove_me, save_results)
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
  distance_metric = 'Euclidean';
  % distance_metric = 'Mahalanobis';

  [~, experiments] = setupExperimentsUsingProjectedImbds(dataset, posneg_balance, true, false);
  % keyboard

  assertion_failure_string = 'because we are plotting ratio of distances (post-projection to pre-projection), we need the original imdb, and at least 1 projected imdb';
  assert(numel(experiments) >= 2, assertion_failure_string);
  assert(numel(strfind(experiments{1}.title, 'Original IMDB')) >= 1, assertion_failure_string);


  figure,
  vectorized_original_imdb = getVectorizedImdb(experiments{1}.imdb);
  for k = 2 : numel(experiments)
    vectorized_projected_imdb = getVectorizedImdb(experiments{k}.imdb);
    assert(length(vectorized_original_imdb.images.labels) == length(vectorized_projected_imdb.images.labels));
    number_of_samples = length(vectorized_original_imdb.images.labels);
    number_of_pairs = number_of_samples * (number_of_samples - 1) / 2;

    % TODO... do any of the RPs change ordering of data points???????

    squareform_pdist_original_imdb = getSquareformPdist(vectorized_original_imdb, distance_metric);
    squareform_pdist_projected_imdb = getSquareformPdist(vectorized_projected_imdb, distance_metric);

    between_class_distances_original_imdb = [];
    between_class_distances_projected_imdb = [];
    within_class_distances_original_imdb = [];
    within_class_distances_projected_imdb = [];


    for i = 1 : number_of_samples
      for j = i + 1 : number_of_samples % don't start loop from index i, because we want to ignore diagonals (which should be 0)
        assert(squareform_pdist_original_imdb(i,j) == squareform_pdist_original_imdb(j,i));
        assert(squareform_pdist_projected_imdb(i,j) == squareform_pdist_projected_imdb(j,i));
        assert(vectorized_original_imdb.images.labels(i) == vectorized_projected_imdb.images.labels(i));
        assert(vectorized_original_imdb.images.labels(j) == vectorized_projected_imdb.images.labels(j));

        label_1 = vectorized_original_imdb.images.labels(i);
        label_2 = vectorized_original_imdb.images.labels(j);
        if label_1 == label_2
          within_class_distances_original_imdb(end+1) = squareform_pdist_original_imdb(i,j);
          within_class_distances_projected_imdb(end+1) = squareform_pdist_projected_imdb(i,j);
        else
          between_class_distances_original_imdb(end+1) = squareform_pdist_original_imdb(i,j);
          between_class_distances_projected_imdb(end+1) = squareform_pdist_projected_imdb(i,j);
        end

        assert(length(within_class_distances_original_imdb) == length(within_class_distances_projected_imdb));
        assert(length(between_class_distances_original_imdb) == length(between_class_distances_projected_imdb));
      end
    end

    subplot(numel(experiments) - 1, 2, 1 + 2 * (k - 2)),
    subplotHistogramOfWithinAndBetweenDistances(within_class_distances_original_imdb, between_class_distances_original_imdb, sprintf('Normalized Distances - %s', experiments{1}.title));

    subplot(numel(experiments) - 1, 2, 2 + 2 * (k - 2)),
    subplotHistogramOfWithinAndBetweenDistances(within_class_distances_projected_imdb, between_class_distances_projected_imdb, sprintf('Normalized Distances - %s', experiments{k}.title));

    % keyboard
    % ratio_of_distances = pdist_original_imdb ./ pdist_projected_imdb;
    % subplot(1, numel(experiments) - 1, k - 1),
    % subplot_title = (sprintf('Ratio of Distances - %s to Original', experiments{k}.title));
    % subplotBeef(ratio_of_distances, subplot_title)
  end

% -------------------------------------------------------------------------
function distance_matrix = getSquareformPdist(imdb, distance_metric)
% -------------------------------------------------------------------------
  switch distance_metric
    case 'Euclidean'
      distance_matrix = squareform(pdist(imdb.images.data));
    case 'Mahalanobis'
      distance_matrix = getSquareformPdistMahalanobis(imdb);
    otherwise
      throwException('[ERROR] Euclidean not recognized.');
  end


% -------------------------------------------------------------------------
function distance_matrix_mahalanobis = getSquareformPdistMahalanobis(vectorized_imdb)
% -------------------------------------------------------------------------
  % imdb is already vectorized, do not vectorize again!
  number_of_samples = size(vectorized_imdb.images.data, 4);
  number_of_classes = numel(unique(vectorized_imdb.images.labels));
  distance_matrix_mahalanobis = zeros(number_of_samples, number_of_samples);

  mean_vectors = {};
  covariance_matrices = {};

  for i = 1 : number_of_classes
    data_for_class = vectorized_imdb.images.data(vectorized_imdb.images.labels == i, :);
    mean_vectors.(sprintf('class_%d', i)) = mean(data_for_class);
    covariance_matrices.(sprintf('class_%d', i)) = cov(data_for_class);
  end


  for i = 1 : number_of_samples
    for j = i + 1 : number_of_samples

    end
  end





% -------------------------------------------------------------------------
function subplotBeef(data, subplot_title)
% -------------------------------------------------------------------------
  x_ticks = [0:0.05:2] * 1;
  y_limits = [0 numel(data)];

  histogram( ...
    data, ...
    x_ticks, ...
    'facealpha', ...
    0.4);

  % title(subplot_title);
  title(sprintf('%.3f +/- %.3f', mean(data), std(data)));

  ylim(y_limits);



% -------------------------------------------------------------------------
function subplotHistogramOfWithinAndBetweenDistances(within_class_distances, between_class_distances, title_string)
% -------------------------------------------------------------------------
  color_palette = {'g', 'b', 'c', 'r', 'k'};
  % legend_entries = {};
  % for i = 1 : numel(experiments)
  %   legend_entries{i} = sprintf('%s - metric = %.4f', experiments{i}.title, experiments{i}.class_metric);
  % end

  all_distances = cat(2, within_class_distances, between_class_distances);
  max_distance = max(all_distances);
  min_distance = min(all_distances);

  normalized_within_class_distances = (within_class_distances - min_distance) / (max_distance - min_distance);
  normalized_between_class_distances = (between_class_distances - min_distance) / (max_distance - min_distance);

  % normalized_within_class_distances = within_class_distances;
  % normalized_between_class_distances = between_class_distances;

  legend_entries = { ...
    sprintf('within dist - %.3f +/- %.3f', mean(normalized_within_class_distances), std(normalized_within_class_distances)),  ...
    sprintf('between dist - %.3f +/- %.3f', mean(normalized_between_class_distances), std(normalized_between_class_distances))};

  x_ticks = 0:0.025:1;
  % x_ticks = 0:100:10000;
  % x_ticks = 0:250:25000;
  % y_limits = [0, numel(all_distances)];
  % y_limits = [0, 1] / 3;
  % y_limits = [0, 0.1];
  y_limits = [0, 0.3];


  hold on
  histogram( ...
    normalized_within_class_distances, ...
    x_ticks, ...
    'Normalization', ...
    'probability', ...
    'facecolor', ...
    color_palette{1}, ...
    'facealpha', ...
    0.4);
  histogram( ...
    normalized_between_class_distances, ...
    x_ticks, ...
    'Normalization', ...
    'probability', ...
    'facecolor', ...
    color_palette{2}, ...
    'facealpha', ...
    0.4);
  % xlim([0, 7000]);
  % ylim([0, 0.05]);
  ylim(y_limits);
  title(title_string);
  hold off
  legend(legend_entries);


































