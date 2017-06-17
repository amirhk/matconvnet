% -------------------------------------------------------------------------
function calculateDistances()
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
  dataset = 'gaussian-5D-160-train-40-test';
  dataset = 'gaussian-10D-160-train-40-test';
  dataset = 'gaussian-25D-160-train-40-test';
  % dataset = 'gaussian-50D-160-train-40-test';
  posneg_balance = 'balanced-38';

  [original_imdb, experiments] = setupExperimentsUsingProjectedImbds(dataset, posneg_balance, 1);

  if true
    % point_type = 'border';
    point_type = 'random';
    % distance_type = 'euclidean';
    distance_type = 'cosine';
    for i = 1 : numel(experiments)
      projected_imdb = experiments{i}.imdb;
      between_class_distance_absolute_values = getPointDistanceAbsoluteValues(experiments{i}.imdb, point_type, distance_type, 'between');
      within_class_distance_absolute_values = getPointDistanceAbsoluteValues(experiments{i}.imdb, point_type, distance_type, 'within');
      experiments{i}.between_class_distance_absolute_values = between_class_distance_absolute_values;
      experiments{i}.within_class_distance_absolute_values = within_class_distance_absolute_values;
    end

    % -------------------------------------------------------------------------
    %                                                                      Plot
    % -------------------------------------------------------------------------
    h = figure;
    color_palette = {'c', 'r', 'g', 'b', 'k'};
    legend_entries = {};
    for i = 1 : numel(experiments)
      legend_entries{i} = experiments{i}.title;
    end

    subplot(1,2,1)
    title(sprintf('Between-class %s Distances', distance_type))
    hold on
    for i = 1 : numel(experiments)
      histogram( ...
        experiments{i}.between_class_distance_absolute_values, ...
        0:2.5:180, ...
        ..., % 0:1:75, ...
        ..., % 0:2.5e-8:10e-7, ...
        'facecolor', ...
        color_palette{mod(i - 1,numel(color_palette)) + 1}, ...
        'facealpha', ...
        0.4);
        % ..., % 'edgecolor', ...
        % ..., % 'none');
    end
    hold off
    legend(legend_entries);


    subplot(1,2,2)
    title(sprintf('Within-class %s Distances', distance_type))
    hold on
    for i = 1 : numel(experiments)
      histogram( ...
        experiments{i}.within_class_distance_absolute_values, ...
        0:2.5:180, ...
        ..., % 0:1:75, ...
        ..., % 0:2.5e-8:10e-7, ...
        'facecolor', ...
        color_palette{mod(i - 1,numel(color_palette)) + 1}, ...
        'facealpha', ...
        0.4);
        % ..., % 'edgecolor', ...
        % ..., % 'none');
    end
    hold off
    legend(legend_entries);

    tmp_string = sprintf('%s %s distances - %s %s - %s', distance_type, point_type, dataset, posneg_balance, experiments{i}.title);
    suptitle(tmp_string);
    saveas(h, fullfile(getDevPath(), 'temp_images', sprintf('%s.png', tmp_string)));

  else

    % -------------------------------------------------------------------------
    %                                                                Get ratios
    % -------------------------------------------------------------------------

    point_type = 'border';
    % point_type = 'random';
    % distance_type = 'euclidean';
    distance_type = 'cosine';
    for i = 1 : numel(experiments)
      projected_imdb = experiments{i}.imdb;
      between_class_distance_ratios = getPointDistanceRatios(original_imdb, projected_imdb, point_type, distance_type, 'between');
      within_class_distance_ratios = getPointDistanceRatios(original_imdb, projected_imdb, point_type, distance_type, 'within');
      experiments{i}.between_class_distance_ratios = between_class_distance_ratios;
      experiments{i}.within_class_distance_ratios = within_class_distance_ratios;
    end

    % -------------------------------------------------------------------------
    %                                                                      Plot
    % -------------------------------------------------------------------------
    figure
    color_palette = {'c', 'r', 'g', 'b', 'k'};
    legend_entries = {};
    for i = 1 : numel(experiments)
      legend_entries{i} = experiments{i}.title;
    end

    subplot(1,2,1)
    title(sprintf('Between-class %s Distance R, distance_typeatios'))
    hold on
    for i = 1 : numel(experiments)
      histogram( ...
        experiments{i}.between_class_distance_ratios, ...
        0:0.05:2.5, ...
        'facecolor', ...
        color_palette{mod(i - 1,numel(color_palette)) + 1}, ...
        'facealpha', ...
        0.4, ...
        'edgecolor', ...
        'none');
    end
    hold off
    legend(legend_entries);


    subplot(1,2,2)
    title(sprintf('Within-class %s Distance R, distance_typeatios'))
    hold on
    for i = 1 : numel(experiments)
      histogram( ...
        experiments{i}.within_class_distance_ratios, ...
        0:0.05:2.5, ...
        'facecolor', ...
        color_palette{mod(i - 1,numel(color_palette)) + 1}, ...
        'facealpha', ...
        0.4, ...
        'edgecolor', ...
        'none');
    end
    hold off
    legend(legend_entries);

    suptitle(sprintf('%s points', point_type));
  end
  % keyboard

  % jigar tala



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
  [original_pdist_matrix, original_labels_train] = getDistanceMatrixAndLabels(original_imdb, distance_type);
  [projected_pdist_matrix, projected_labels_train] = getDistanceMatrixAndLabels(projected_imdb, distance_type);

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
      repeat_count = 10;
    otherwise
      throwException('[ERROR] point_type not recognized.');
  end
  distance_ratios = [];
  distance_absolute_values = [];
  for i = 1 : repeat_count
    for point_index = 1 : size(original_pdist_matrix, 1)

      % -------------------------------------------------------------------------
      %                                                         Get point indices
      % -------------------------------------------------------------------------
      original_point_row = original_pdist_matrix(point_index, :);
      original_point_class = original_labels_train(point_index);
      projected_point_row = projected_pdist_matrix(point_index, :);
      projected_point_class = projected_labels_train(point_index);

      if strcmp(point_type, 'border')
        original_point_index = findBorderPointFunctionHandle(original_point_row, original_point_class, original_labels_train);
        projected_point_index = findBorderPointFunctionHandle(projected_point_row, projected_point_class, projected_labels_train);
      elseif strcmp(point_type, 'random')
        original_point_index = findRandomPointFunctionHandle(original_point_row, original_point_class, original_labels_train);
        projected_point_index = original_point_index;
      end

      % -------------------------------------------------------------------------
      %                                                     Get points themselves
      % -------------------------------------------------------------------------
      original_reference_point_index = point_index;
      original_reference_point = getVectorizedSampleAtIndex(original_imdb, original_reference_point_index);
      original_point = getVectorizedSampleAtIndex(original_imdb, original_point_index);

      projected_reference_point_index = point_index;
      projected_reference_point = getVectorizedSampleAtIndex(projected_imdb, projected_reference_point_index);
      projected_point = getVectorizedSampleAtIndex(projected_imdb, projected_point_index);

      % -------------------------------------------------------------------------
      %                                                        Calculate distance
      % -------------------------------------------------------------------------
      original_distance = calculateDistance(original_reference_point, original_reference_point_index, original_point, original_point_index, distance_type, original_pdist_matrix);
      projected_distance = calculateDistance(projected_reference_point, projected_reference_point_index, projected_point, projected_point_index, distance_type, projected_pdist_matrix);

      % -------------------------------------------------------------------------
      %                                                 Finally, calculate ratios
      % -------------------------------------------------------------------------
      distance_ratios(end + 1) = projected_distance / original_distance;
      distance_absolute_values(end + 1) = original_distance;

    end
  end


% -------------------------------------------------------------------------
function [matrix_pdist, labels_train] = getDistanceMatrixAndLabels(imdb, distance_type)
% -------------------------------------------------------------------------
  data_train = imdb.images.data(:,:,:,imdb.images.set == 1);
  labels_train = imdb.images.labels(imdb.images.set == 1);
  sample_size = size(data_train, 1) * size(data_train, 2) * size(data_train, 3);
  samples = reshape(data_train, sample_size, [])';
  % keyboard
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






































