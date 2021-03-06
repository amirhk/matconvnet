% -------------------------------------------------------------------------
function tmpScriptCalculateDistances(dataset, posneg_balance, save_results)
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
  afprintf(sprintf('[INFO] Setting up experiment...\n'));
  [original_imdb, experiments] = setupExperimentsUsingProjectedImbds(dataset, posneg_balance, true, false);
  afprintf(sprintf('[INFO] done!\n'));
  printConsoleOutputSeparator();

  plot_type = 'absolute_distances';
  % plot_type = 'ratio_distances_giryes_paper';
  % plot_type = 'ratio_distances_discuss_w_alex';

  if strcmp(plot_type, 'absolute_distances')

    % -------------------------------------------------------------------------
    %                                                    Get absolute distances
    % -------------------------------------------------------------------------
    afprintf(sprintf('[INFO] Get absolute distances...\n'));
    % point_type = 'border';
    point_type = 'random';
    % distance_type = 'euclidean';
    distance_type = 'cosine';

    for i = 1 : numel(experiments)
      projected_imdb = experiments{i}.imdb;
      experiments{i}.between_class_distance_absolute_values = getPointDistanceAbsoluteValues(experiments{i}.imdb, point_type, distance_type, 'between');
      experiments{i}.within_class_distance_absolute_values = getPointDistanceAbsoluteValues(experiments{i}.imdb, point_type, distance_type, 'within');
    end

    afprintf(sprintf('[INFO] done!\n'));
    printConsoleOutputSeparator();


    % -------------------------------------------------------------------------
    %                                                                      Plot
    % -------------------------------------------------------------------------
    afprintf(sprintf('[INFO] Plotting...\n'));
    if numel(experiments) == 2
      h = figure;

      subplot(1,2,1),
      within_between = 'between';
      subplotBeefAbsoluteDistance(experiments, within_between, distance_type);

      subplot(1,2,2),
      within_between = 'within';
      subplotBeefAbsoluteDistance(experiments, within_between, distance_type);

    else
      assert(numel(experiments) == 14);
      tmp = struct();
      tmp.(sprintf('group_%d', 1)) = cat(2, experiments(1), experiments(2:4));
      tmp.(sprintf('group_%d', 2)) = cat(2, experiments(1), experiments(5:7));
      tmp.(sprintf('group_%d', 3)) = cat(2, experiments(8), experiments(9:11));
      tmp.(sprintf('group_%d', 4)) = cat(2, experiments(8), experiments(12:14));

      h = figure;
      for k = 1 : 4
        experiments = tmp.(sprintf('group_%d', k));

        subplot(4, 2, 1 + (k - 1) * 2)
        within_between = 'between';
        subplotBeefAbsoluteDistance(experiments, within_between, distance_type);

        subplot(4, 2, 2 + (k - 1) * 2)
        within_between = 'within';
        subplotBeefAbsoluteDistance(experiments, within_between, distance_type);

      end
    end
    afprintf(sprintf('[INFO] done!\n'));
    printConsoleOutputSeparator();

    tmp_string = sprintf('%s %s distances - %s - %s', distance_type, point_type, dataset, posneg_balance);
    suptitle(tmp_string);

  elseif strcmp(plot_type, 'ratio_distances_giryes_paper')

    % -------------------------------------------------------------------------
    %                                                    Get ratio of distances
    % -------------------------------------------------------------------------
    afprintf(sprintf('[INFO] Get ratio of distances...\n'));
    point_type = 'border';
    % point_type = 'random';
    % distance_type = 'euclidean';
    distance_type = 'cosine';

    for i = 1 : numel(experiments)
      projected_imdb = experiments{i}.imdb;
      experiments{i}.between_class_distance_ratios = getPointDistanceRatios(original_imdb, projected_imdb, point_type, distance_type, 'between');
      experiments{i}.within_class_distance_ratios = getPointDistanceRatios(original_imdb, projected_imdb, point_type, distance_type, 'within');
    end
    afprintf(sprintf('[INFO] done!\n'));
    printConsoleOutputSeparator();


    % -------------------------------------------------------------------------
    %                                                                      Plot
    % -------------------------------------------------------------------------
    afprintf(sprintf('[INFO] Plotting...\n'));
    h = figure;

    subplot(1,2,1)
    within_between = 'between';
    subplotBeefGiryesRatioDistance(experiments, within_between, distance_type);

    subplot(1,2,2)
    within_between = 'within';
    subplotBeefGiryesRatioDistance(experiments, within_between, distance_type);

    suptitle(sprintf('%s points', point_type));

    afprintf(sprintf('[INFO] done!\n'));
    printConsoleOutputSeparator();


  elseif strcmp(plot_type, 'ratio_distances_discuss_w_alex')

    % -------------------------------------------------------------------------
    %                                                    Get ratio of distances
    % -------------------------------------------------------------------------
    afprintf(sprintf('[INFO] Get ratio of distances...\n'));
    distance_type = 'euclidean';
    % distance_type = 'cosine';
    for i = 1 : numel(experiments)
      projected_imdb = experiments{i}.imdb;
      experiments{i}.between_class_to_within_class_distance_ratios = getBetweenToWithinDistanceRatios(experiments{i}.imdb, distance_type);
      experiments{i}.fisher_discriminant_ratio = getFisherDiscriminantRatio(experiments{i}.imdb);
    end

    afprintf(sprintf('[INFO] done!\n'));
    printConsoleOutputSeparator();

    % -------------------------------------------------------------------------
    %                                                                      Plot
    % -------------------------------------------------------------------------
    afprintf(sprintf('[INFO] Plotting...\n'));
    h = figure;
    plotBeefAlexRatioDistance(experiments);

    afprintf(sprintf('[INFO] done!\n'));
    printConsoleOutputSeparator();

  end

  if save_results
    print(fullfile(getDevPath(), 'temp_images', tmp_string), '-dpdf', '-fillpage')
  end


% -------------------------------------------------------------------------
function ratio_distances = getBetweenToWithinDistanceRatios(imdb, distance_type);
% -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] Getting between class to within class distance ratios...\n'));
  [pdist_matrix, labels] = getDistanceMatrixAndLabels(imdb);
  ratio_distances = [];
  for i = 1:size(pdist_matrix, 1)
    same_class_distance = [];
    diff_class_distance = [];
    for j = 1:size(pdist_matrix, 2)
      if labels(i) == labels(j)
        same_class_distance(end + 1) = pdist_matrix(i, j);
      else
        diff_class_distance(end + 1) = pdist_matrix(i, j);
      end
    end
    ratio_distances(end + 1) = mean(diff_class_distance) / mean(same_class_distance);
  end
  afprintf(sprintf('[INFO] done!\n'));


% -------------------------------------------------------------------------
function distance_absolute_values = getPointDistanceAbsoluteValues(imdb, point_type, distance_type, within_between)
% -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] Getting point distance ratios...\n'));
  [~, distance_absolute_values] = getPointDistanceBeef(imdb, imdb, point_type, distance_type, within_between, 'none');
  afprintf(sprintf('[INFO] done!\n'));


% -------------------------------------------------------------------------
function distance_ratios = getPointDistanceRatios(original_imdb, projected_imdb, point_type, distance_type, within_between)
% -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] Getting point distance ratios...\n'));
  [distance_ratios, ~] = getPointDistanceBeef(original_imdb, projected_imdb, point_type, distance_type, within_between, 'sum-normalized');
  afprintf(sprintf('[INFO] done!\n'));


% -------------------------------------------------------------------------
function [distance_ratios, distance_absolute_values] = getPointDistanceBeef(original_imdb, projected_imdb, point_type, distance_type, within_between, normalization_type)
% -------------------------------------------------------------------------
  [original_pdist_matrix, original_labels_train] = getDistanceMatrixAndLabels(original_imdb);
  [projected_pdist_matrix, projected_labels_train] = getDistanceMatrixAndLabels(projected_imdb);

  [original_pdist_matrix, projected_pdist_matrix] = normalizePdistMatrices(original_pdist_matrix, projected_pdist_matrix, normalization_type);

  assert(size(original_pdist_matrix, 1) == size(projected_pdist_matrix, 1));
  assert(size(original_pdist_matrix, 2) == size(projected_pdist_matrix, 2));

  % -------------------------------------------------------------------------
  %                                                  Define function handles
  % -------------------------------------------------------------------------
  if strcmp(within_between, 'within')
    findBorderPointFunctionHandle = @findFurthestWithinClassPointIndexToPoint;
    findRandomPointFunctionHandle = @findRandomWithinClassPointIndexToPoint;
  elseif strcmp(within_between, 'between')
    findBorderPointFunctionHandle = @findClosestBetweenClassPointIndexToPoint;
    findRandomPointFunctionHandle = @findRandomBetweenClassPointIndexToPoint;
  else
    throwException('[ERROR] within_between not recognized.');
  end

  % -------------------------------------------------------------------------
  %                                                  Get ratios of all points
  % -------------------------------------------------------------------------
  switch point_type
    case 'border'
      repeat_count = 1;
    case 'random'
      repeat_count = 100;
    otherwise
      throwException('[ERROR] point_type not recognized.');
  end
  distance_ratios = [];
  distance_absolute_values = [];
  for i = 1 : repeat_count
    for reference_point_index = 1 : size(original_pdist_matrix, 1)

      % -------------------------------------------------------------------------
      %                                                         Get point indices
      % -------------------------------------------------------------------------
      original_reference_point_row = original_pdist_matrix(reference_point_index, :);
      original_reference_point_class = original_labels_train(reference_point_index);
      projected_reference_point_row = projected_pdist_matrix(reference_point_index, :);
      projected_reference_point_class = projected_labels_train(reference_point_index);
      assert(original_reference_point_class == projected_reference_point_class);

      if strcmp(point_type, 'border')
        original_other_point_index = findBorderPointFunctionHandle(original_reference_point_row, original_reference_point_class, original_labels_train);
        projected_other_point_index = findBorderPointFunctionHandle(projected_reference_point_row, projected_reference_point_class, projected_labels_train);
      elseif strcmp(point_type, 'random')
        original_other_point_index = findRandomPointFunctionHandle(original_reference_point_row, original_reference_point_class, original_labels_train);
        projected_other_point_index = original_other_point_index;
      end

      % -------------------------------------------------------------------------
      %                                                     Get points themselves
      % -------------------------------------------------------------------------
      original_reference_point_index = reference_point_index;
      original_reference_point = getVectorizedSampleAtIndex(original_imdb, original_reference_point_index);
      original_other_point = getVectorizedSampleAtIndex(original_imdb, original_other_point_index);

      projected_reference_point_index = reference_point_index;
      projected_reference_point = getVectorizedSampleAtIndex(projected_imdb, projected_reference_point_index);
      projected_other_point = getVectorizedSampleAtIndex(projected_imdb, projected_other_point_index);

      % -------------------------------------------------------------------------
      %                                                        Calculate distance
      % -------------------------------------------------------------------------
      original_distance = calculateDistance(original_reference_point, original_reference_point_index, original_other_point, original_other_point_index, distance_type, original_pdist_matrix);
      projected_distance = calculateDistance(projected_reference_point, projected_reference_point_index, projected_other_point, projected_other_point_index, distance_type, projected_pdist_matrix);

      % -------------------------------------------------------------------------
      %                                                 Finally, calculate ratios
      % -------------------------------------------------------------------------
      distance_ratios(end + 1) = projected_distance / original_distance;
      distance_absolute_values(end + 1) = original_distance;

    end
  end


% -------------------------------------------------------------------------
function [matrix_pdist, labels_train] = getDistanceMatrixAndLabels(imdb)
% -------------------------------------------------------------------------
  data_train = imdb.images.data(:,:,:,imdb.images.set == 1);
  labels_train = imdb.images.labels(imdb.images.set == 1);
  sample_size = size(data_train, 1) * size(data_train, 2) * size(data_train, 3);
  samples = reshape(data_train, sample_size, [])';
  matrix_pdist = squareform(pdist(samples));


% -------------------------------------------------------------------------
function distance = calculateDistance(point_1, point_1_index, point_2, point_2_index, distance_type, pdist_matrix)
% -------------------------------------------------------------------------
  if strcmp(distance_type, 'euclidean')
    % TODO: these distances are sadly not normalized :|
    % distance = pdist([point_1; point_2]);
    assert(pdist_matrix(point_1_index, point_2_index) == pdist_matrix(point_2_index, point_1_index));
    distance = pdist_matrix(point_1_index, point_2_index);
  elseif strcmp(distance_type, 'cosine')
    cos_theta = dot(point_1, point_2) / (norm(point_1) * norm(point_2));
    theta_in_degrees = acosd(cos_theta);
    if ~isreal(theta_in_degrees)
      % keyboard
      % TODO?????????????
      theta_in_degrees = real(theta_in_degrees);
    end
    distance = theta_in_degrees;
  end


% -------------------------------------------------------------------------
function point_index = findClosestBetweenClassPointIndexToPoint(point_row, point_class, labels)
% -------------------------------------------------------------------------
  % [sorted_point_row, indices] = sort(point_row(labels ~= point_class));
  % point_index = indices(1);

  [sorted_point_row, indices] = sort(point_row, 'ascend');
  labels = labels(indices); % sort labels according to sorted_point_row
  for i = 1:length(point_row)
    if labels(i) ~= point_class
      point_index = indices(i);
      break
    end
  end


% -------------------------------------------------------------------------
function point_index = findFurthestWithinClassPointIndexToPoint(point_row, point_class, labels)
% -------------------------------------------------------------------------
  % [sorted_point_row, indices] = sort(point_row(labels == point_class));
  % point_index = indices(end);

  [sorted_point_row, indices] = sort(point_row, 'descend');
  labels = labels(indices); % sort labels according to sorted_point_row
  for i = 1:length(point_row)
    if labels(i) == point_class
      point_index = indices(i);
      break
    end
  end


% -------------------------------------------------------------------------
function random_point_index = findRandomBetweenClassPointIndexToPoint(point_row, point_class, labels)
% -------------------------------------------------------------------------
  % between_class_row = point_row(labels ~= point_class);
  % random_point_index = ceil(rand() * length(between_class_row));

  while 1
    random_point_index = ceil(rand() * length(point_row));
    if labels(random_point_index) ~= point_class
      break
    end
  end


% -------------------------------------------------------------------------
function random_point_index = findRandomWithinClassPointIndexToPoint(point_row, point_class, labels)
% -------------------------------------------------------------------------
  % within_class_row = point_row(labels == point_class);
  % random_point_index = ceil(rand() * length(within_class_row));

  while 1
    random_point_index = ceil(rand() * length(point_row));
    if labels(random_point_index) == point_class
      break
    end
  end


% -------------------------------------------------------------------------
function [original_pdist_matrix, projected_pdist_matrix] = normalizePdistMatrices(original_pdist_matrix, projected_pdist_matrix, normalization_type)
% -------------------------------------------------------------------------
  max_o = max(original_pdist_matrix(:));
  min_o = min(original_pdist_matrix(:));
  sum_o = sum(original_pdist_matrix(:)) / 2;
  max_p = max(projected_pdist_matrix(:));
  min_p = min(projected_pdist_matrix(:));
  sum_p = sum(projected_pdist_matrix(:)) / 2;

  switch normalization_type

    case 'max-normalized'
      % MAX-NORMALIZED
      original_pdist_matrix = (original_pdist_matrix - min_o) / (max_o - min_o);
      projected_pdist_matrix = (projected_pdist_matrix - min_p) / (max_p - min_p);

    case 'sum-normalized'
      % SUM-NORMALIZED
      original_pdist_matrix = original_pdist_matrix / sum_o; % remember pdist is symmetric
      projected_pdist_matrix = projected_pdist_matrix / sum_p; % remember pdist is symmetric

    case 'email-max-normalized'
      % EMAIL MAX-NORMALIZED (NORMALIZE PROJECTED SPACE INTO DYNAMIC RANGE OF INPUT SPACE)
      original_pdist_matrix = original_pdist_matrix;
      projected_pdist_matrix = (projected_pdist_matrix - min_p) / (max_p - min_p) * (max_o - min_o) + min_o;

    case 'email-sum-normalized'
      % EMAIL SUM-NORMALIZED
      original_pdist_matrix = original_pdist_matrix / sum_o; % remember pdist is symmetric
      projected_pdist_matrix = projected_pdist_matrix / sum_p * sum_o; % remember pdist is symmetric

    case 'none'
      original_pdist_matrix = original_pdist_matrix;
      projected_pdist_matrix = projected_pdist_matrix;

    otherwise
      throwException('[ERROR] normalization_type not recognized.');

  end


% -------------------------------------------------------------------------
function subplotBeefAbsoluteDistance(experiments, within_between, distance_type)
% -------------------------------------------------------------------------
  color_palette = {'c', 'r', 'g', 'b', 'k'};
  legend_entries = {};
  for i = 1 : numel(experiments)
    legend_entries{i} = experiments{i}.title;
  end

  if strcmp(distance_type, 'cosine')
    x_ticks = 0:2.5:180;
    y_limits = [0 15000];
  else
    x_ticks = 0:0.1:10;
    y_limits = [0 15000];
  end

  if strcmp(within_between, 'within')
    title(sprintf('Within-class %s Distances', distance_type));
  else
    title(sprintf('Between-class %s Distances', distance_type));
  end

  hold on
  for i = 1 : numel(experiments)
    if strcmp(within_between, 'within')
      data = experiments{i}.within_class_distance_absolute_values;
    else
      data = experiments{i}.between_class_distance_absolute_values;
    end
    histogram( ...
      data, ...
      x_ticks, ...
      'facecolor', ...
      color_palette{mod(i - 1,numel(color_palette)) + 1}, ...
      'facealpha', ...
      0.4);
  end
  ylim(y_limits);
  hold off
  legend(legend_entries);


% -------------------------------------------------------------------------
function subplotBeefGiryesRatioDistance(experiments, within_between, distance_type)
% -------------------------------------------------------------------------
  color_palette = {'c', 'r', 'g', 'b', 'k'};
  legend_entries = {};
  for i = 1 : numel(experiments)
    legend_entries{i} = experiments{i}.title;
  end


  if strcmp(distance_type, 'cosine')
    x_ticks = 0:2.5:180;
    y_limits = [0 30000];
  else
    x_ticks = 0:0.1:10;
    y_limits = [0 30000];
  end

  if strcmp(within_between, 'within')
    title(sprintf('Within-class %s Distance Ratios, distance_type'));
  else
    title(sprintf('Between-class %s Distance Ratios, distance_type'));
  end

  hold on
  for i = 1 : numel(experiments)
    if strcmp(within_between, 'within')
      data = experiments{i}.within_class_distance_ratios;
    else
      data = experiments{i}.between_class_distance_ratios;
    end
    histogram( ...
      data, ...
      x_ticks, ...
      'facecolor', ...
      color_palette{mod(i - 1,numel(color_palette)) + 1}, ...
      'facealpha', ...
      0.4);
  end
  ylim(y_limits);
  hold off
  legend(legend_entries);


% -------------------------------------------------------------------------
function plotBeefAlexRatioDistance(experiments)
% -------------------------------------------------------------------------
  color_palette = {'c', 'r', 'g', 'b', 'k'};
  legend_entries = {};
  for i = 1 : numel(experiments)
    legend_entries{i} = sprintf('%s - fd = %.3f', experiments{i}.title, experiments{i}.fisher_discriminant_ratio);
    % legend_entries{i} = experiments{i}.title;
  end

  x_ticks = 0:0.025:3;
  y_limits = [0 100];
  title('Between-class to Within-class Distance Ratios');

  hold on
  for i = 1 : numel(experiments)
    data = experiments{i}.between_class_to_within_class_distance_ratios;
    histogram( ...
      data, ...
      x_ticks, ...
      'facecolor', ...
      color_palette{mod(i - 1,numel(color_palette)) + 1}, ...
      'facealpha', ...
      0.4);
  end
  ylim(y_limits);
  hold off
  legend(legend_entries);



% -------------------------------------------------------------------------
function fisher_discriminant_ratio = getFisherDiscriminantRatio(imdb)
% -------------------------------------------------------------------------
  vectorized_imdb = getVectorizedImdb(imdb);
  assert(numel(unique(imdb.images.labels)) == 2);

  data = vectorized_imdb.images.data;
  data_class_1 = data(vectorized_imdb.images.labels == 1, :);
  data_class_2 = data(vectorized_imdb.images.labels == 2, :);

  mean_data_class_1 = mean(data_class_1);
  mean_data_class_2 = mean(data_class_2);

  cov_data_class_1 = cov(data_class_1);
  cov_data_class_2 = cov(data_class_2);

  % Using Alex's suggested method
  % fisher_discriminant_ratio = norm(mean_data_class_1 - mean_data_class_2)^2 / (norm(cov_data_class_1 + cov_data_class_2));

  % Using http://www.csd.uwo.ca/~olga/Courses/CS434a_541a/Lecture8.pdf
  % S_1 = size(data_class_1, 1) * cov_data_class_1;
  S_1 = cov_data_class_1;
  % S_2 = size(data_class_2, 1) * cov_data_class_2;
  S_2 = cov_data_class_2;

  S_W = S_1 + S_2;

  % assert full rank
  assert(rank(S_W) == size(S_W, 1));

  optimal_projection_line_direction = inv(S_W) * (mean_data_class_1 - mean_data_class_2)';

  optimally_projected_data = optimal_projection_line_direction' * data';
  optimally_projected_data_class_1 = optimal_projection_line_direction' * data_class_1';
  optimally_projected_data_class_2 = optimal_projection_line_direction' * data_class_2';

  % Using https://compbio.soe.ucsc.edu/genex/genexTR2html/node12.html
  fisher_discriminant_ratio = ...
    (mean(optimally_projected_data_class_1) - mean(optimally_projected_data_class_2))^2 / ...
    (var(optimally_projected_data_class_1) + var(optimally_projected_data_class_2));

  % keyboard
  % figure, hold on, histogram(optimally_projected_data_class_1), histogram(optimally_projected_data_class_2)
  % imdb.images.data = optimally_projected_data;
  % imdb = get4DImdb(imdb, size(imdb.images.data, 1), 1, 1, []);





















































