% -------------------------------------------------------------------------
function dumb_array = tmpScriptCalculateDistances(dataset, posneg_balance, save_results)
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
  dumb_array = {};

  [~, experiments] = setupExperimentsUsingProjectedImbds(dataset, posneg_balance, 1);
  for i = 1 : numel(experiments)
    dumb_array.(sprintf('exp_%d_name', i)) = experiments{i}.title;
  end
  for distance_type = {'euclidean', 'cosine'}
    distance_type = char(distance_type);
    for within_between = {'between', 'within'}
      within_between = char(within_between);
      for i = 1 : numel(experiments)
        dumb_array.(sprintf('exp_%d_name', i)) = experiments{i}.title;
        dumb_array.(sprintf('exp_%d_%s_%s_metric', i, distance_type, within_between)) = [];
      end
    end
  end

  for kkk = 1:10
    afprintf(sprintf('[INFO] Setting up experiment...\n'));
    [original_imdb, experiments] = setupExperimentsUsingProjectedImbds(dataset, posneg_balance, 1);


    % % keyboard
    % for i = 1 : numel(experiments)
    %   tmp = experiments{i}.imdb;
    %   tmp.images.data = tmp.images.data(:,:,:,1:5);
    %   tmp.images.labels = tmp.images.labels(1:5);
    %   tmp.images.set = tmp.images.set(1:5);
    %   experiments{i}.imdb = tmp;



    %   tmp = getVectorizedImdb(experiments{i}.imdb);
    %   % [sorted_labels, sorted_indices] = sort(tmp.images.labels);
    %   % sorted_data = tmp.images.data(sorted_indices, :);
    %   sorted_data = tmp.images.data;
    %   disp(sorted_data);
    %   % keyboard
    % end
    % % keyboard


    afprintf(sprintf('[INFO] done!\n'));
    printConsoleOutputSeparator();

    plot_type = 'absolute_distances';
    % plot_type = 'ratio_distances_giryes_paper';
    % plot_type = 'ratio_distances_discuss_w_alex';

    if strcmp(plot_type, 'absolute_distances')

      % other_point_type = 'border';
      % other_point_type = 'random';
      other_point_type = 'average_of_all';

      h = figure;
      distance_types = {'euclidean', 'cosine'};
      % distance_types = {'euclidean'};
      % distance_types = {'cosine'};
      within_between_types = {'between', 'within'};
      % within_between_types = {'between'};
      for k1 = 1 : numel(distance_types)
        for k2 = 1 : numel(within_between_types)

          distance_type = distance_types{k1};
          within_between = within_between_types{k2};

          % -------------------------------------------------------------------------
          %                                                    Get absolute distances
          % -------------------------------------------------------------------------
          afprintf(sprintf('[INFO] Getting absolute distances...\n'));
          for i = 1 : numel(experiments)
            [experiments{i}.distance_absolute_values, experiments{i}.class_metric] = ...
              getPointDistanceAbsoluteValues(experiments{i}.imdb, other_point_type, distance_type, within_between);


            % experiments{i}.class_metric
            % keyboard

            tmp = dumb_array.(sprintf('exp_%d_%s_%s_metric', i, distance_type, within_between));
            tmp(end+1) = experiments{i}.class_metric;
            dumb_array.(sprintf('exp_%d_%s_%s_metric', i, distance_type, within_between)) = tmp;
          end
          afprintf(sprintf('[INFO] done!\n'));
          printConsoleOutputSeparator();


          % -------------------------------------------------------------------------
          %                                                                      Plot
          % -------------------------------------------------------------------------
          afprintf(sprintf('[INFO] Plotting...\n'));
          subplot(numel(distance_types), numel(within_between_types), 1 + (k2 - 1) + (k1 - 1) * numel(within_between_types)),
          subplotBeefAbsoluteDistance(experiments, within_between, distance_type);
          afprintf(sprintf('[INFO] done!\n'));
          printConsoleOutputSeparator();

        end
      end

      tmp_string = sprintf('%s %s distances - %s - %s', distance_type, other_point_type, dataset, posneg_balance);
      suptitle(tmp_string);

    elseif strcmp(plot_type, 'ratio_distances_giryes_paper')

      % other_point_type = 'border';
      % other_point_type = 'random';
      other_point_type = 'average_of_all';

      h = figure;
      distance_types = {'euclidean', 'cosine'};
      for k = 1 : numel(distance_types)

        distance_type = distance_types{k};

        % -------------------------------------------------------------------------
        %                                                    Get ratio of distances
        % -------------------------------------------------------------------------
        afprintf(sprintf('[INFO] Getting ratio of distances...\n'));
        for i = 1 : numel(experiments)
          projected_imdb = experiments{i}.imdb;
          experiments{i}.between_class_distance_ratios = getPointDistanceRatios(original_imdb, projected_imdb, other_point_type, distance_type, 'between');
          experiments{i}.within_class_distance_ratios = getPointDistanceRatios(original_imdb, projected_imdb, other_point_type, distance_type, 'within');
        end
        afprintf(sprintf('[INFO] done!\n'));
        printConsoleOutputSeparator();

        % -------------------------------------------------------------------------
        %                                                                      Plot
        % -------------------------------------------------------------------------
        afprintf(sprintf('[INFO] Plotting...\n'));

        subplot(numel(distance_types), 2, 1 + (k - 1) * numel(distance_types)),
        within_between = 'between';
        subplotBeefGiryesRatioDistance(experiments, within_between, distance_type);

        subplot(numel(distance_types), 2, 2 + (k - 1) * numel(distance_types)),
        within_between = 'within';
        subplotBeefGiryesRatioDistance(experiments, within_between, distance_type);

        suptitle(sprintf('%s points', other_point_type));

        afprintf(sprintf('[INFO] done!\n'));
        printConsoleOutputSeparator();

      end

    elseif strcmp(plot_type, 'ratio_distances_discuss_w_alex')

      h = figure;
      distance_types = {'euclidean', 'cosine'};
      for k = 1 : numel(distance_types)

        distance_type = distance_types{k};

        % -------------------------------------------------------------------------
        %                                                    Get ratio of distances
        % -------------------------------------------------------------------------
        afprintf(sprintf('[INFO] Getting ratio of distances...\n'));
        distance_type = 'euclidean';
        for i = 1 : numel(experiments)
          [ ...
            experiments{i}.between_class_to_within_class_distance_ratios, ...
            experiments{i}.between_class_metric, ...
            experiments{i}.within_class_metric] = ...
            getBetweenToWithinPointDistanceRatios(experiments{i}.imdb, distance_type);
          experiments{i}.fisher_discriminant_ratio = getFisherDiscriminantRatio(experiments{i}.imdb);
        end

        afprintf(sprintf('[INFO] done!\n'));
        printConsoleOutputSeparator();

        % -------------------------------------------------------------------------
        %                                                                      Plot
        % -------------------------------------------------------------------------
        afprintf(sprintf('[INFO] Plotting...\n'));
        subplot(numel(distance_types), 1, 1 + (k - 1) * numel(distance_types)),
        plotBeefAlexRatioDistance(experiments, distance_type);

        afprintf(sprintf('[INFO] done!\n'));
        printConsoleOutputSeparator();

      end

    end

    save('dumb_array.mat', 'dumb_array');
  end



  if save_results
    print(fullfile(getDevPath(), 'temp_images', tmp_string), '-dpdf', '-fillpage')
  end


% -------------------------------------------------------------------------
function [ratio_distances, between_class_metric, within_class_metric] = getBetweenToWithinPointDistanceRatios(imdb, distance_type);
% -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] Getting between class TO within class distance ratios...\n'));
  [~, within_class_distance_absolute_values, within_class_metric] = getPointDistanceBeef(imdb, imdb, 'average_of_all', distance_type, 'within', 'none');
  [~, between_class_distance_absolute_values, between_class_metric] = getPointDistanceBeef(imdb, imdb, 'average_of_all', distance_type, 'between', 'none');
  assert(length(within_class_distance_absolute_values) == length(between_class_distance_absolute_values));
  % TODO... is it correct to just do the below:
  ratio_distances = between_class_distance_absolute_values ./ within_class_distance_absolute_values;
  afprintf(sprintf('[INFO] done!\n'));


% -------------------------------------------------------------------------
function [distance_absolute_values, class_metric] = getPointDistanceAbsoluteValues(imdb, other_point_type, distance_type, within_between)
% -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] Getting point distance ratios...\n'));
  [~, distance_absolute_values, class_metric] = getPointDistanceBeef(imdb, imdb, other_point_type, distance_type, within_between, 'none');
  afprintf(sprintf('[INFO] done!\n'));


% -------------------------------------------------------------------------
function distance_ratios = getPointDistanceRatios(original_imdb, projected_imdb, other_point_type, distance_type, within_between)
% -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] Getting point distance ratios...\n'));
  [distance_ratios, ~, ~] = getPointDistanceBeef(original_imdb, projected_imdb, other_point_type, distance_type, within_between, 'sum-normalized');
  afprintf(sprintf('[INFO] done!\n'));


% -------------------------------------------------------------------------
function [ ...
  projected_2_original_distance_ratios, ...
  original_distance_absolute_values, ...
  average_of_all_original_distance_absolute_values] = getPointDistanceBeef(original_imdb, projected_imdb, other_point_type, distance_type, within_between, normalization_type)
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
    findBorderPointIndexFunctionHandle = @findFurthestWithinClassPointIndexToPoint;
    findRandomPointIndexFunctionHandle = @findRandomWithinClassPointIndexToPoint;
    findAllOtherPointIndicesFunctionHandle = @findAllOtherWithinClassPointIndicesToPoint;
  elseif strcmp(within_between, 'between')
    findBorderPointIndexFunctionHandle = @findClosestBetweenClassPointIndexToPoint;
    findRandomPointIndexFunctionHandle = @findRandomBetweenClassPointIndexToPoint;
    findAllOtherPointIndicesFunctionHandle = @findAllOtherBetweenClassPointIndicesToPoint;
  else
    throwException('[ERROR] within_between not recognized.');
  end

  % -------------------------------------------------------------------------
  %                                                  Get ratios of all points
  % -------------------------------------------------------------------------
  switch other_point_type
    case 'border'
      repeat_count = 1;
    case 'random'
      repeat_count = 100;
    case 'average_of_all'
      repeat_count = 1;
    otherwise
      throwException('[ERROR] other_point_type not recognized.');
  end
  projected_2_original_distance_ratios = [];
  original_distance_absolute_values = [];
  original_distance_absolute_values_all = [];

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

      original_reference_point_index = reference_point_index;
      projected_reference_point_index = reference_point_index;

      if strcmp(other_point_type, 'border')
        original_other_point_index = findBorderPointIndexFunctionHandle(original_reference_point_row, original_reference_point_class, original_labels_train);
        projected_other_point_index = findBorderPointIndexFunctionHandle(projected_reference_point_row, projected_reference_point_class, projected_labels_train);
      elseif strcmp(other_point_type, 'random')
        original_other_point_index = findRandomPointIndexFunctionHandle(original_reference_point_row, original_reference_point_class, original_labels_train);
        projected_other_point_index = original_other_point_index;
      elseif strcmp(other_point_type, 'average_of_all')
        original_other_point_indices = findAllOtherPointIndicesFunctionHandle(original_reference_point_index, original_reference_point_class, original_labels_train);
        projected_other_point_indices = original_other_point_indices; % turns out to be the same `set` of indices, because the reference point is the same
      end

      % -------------------------------------------------------------------------
      %                                                        Calculate distance
      % -------------------------------------------------------------------------



      % original_reference_point_index
      % original_other_point_indices
      % keyboard



      if ~strcmp(other_point_type, 'average_of_all')
        original_distance = calculateDistance(original_imdb, original_reference_point_index, original_other_point_index, distance_type, original_pdist_matrix);
        projected_distance = calculateDistance(projected_imdb, projected_reference_point_index, projected_other_point_index, distance_type, projected_pdist_matrix);
      else
        tmp_original_distance = [];
        tmp_projected_distance = [];
        for original_other_point_index = original_other_point_indices
          projected_other_point_index = original_other_point_index;
          tmp_original_distance(end+1) = calculateDistance(original_imdb, original_reference_point_index, original_other_point_index, distance_type, original_pdist_matrix);
          tmp_projected_distance(end+1) = calculateDistance(projected_imdb, projected_reference_point_index, projected_other_point_index, distance_type, projected_pdist_matrix);


          % point_1 = original_imdb.images.data(:,:,:,original_reference_point_index);
          % point_2 = original_imdb.images.data(:,:,:,original_other_point_index);
          % point_1
          % point_2
          % % original_pdist_matrix(original_reference_point_index, original_other_point_index)
          % % norm(a - b)
          % % norm(b - a)
          % cos_theta = dot(point_1, point_2) / (norm(point_1) * norm(point_2));
          % acosd(cos_theta)
          % disp('--');
          % tmp_original_distance(end)
          % keyboard
          original_distance_absolute_values_all(end+1) = tmp_original_distance(end);

        end
        tmp_original_distance = tmp_original_distance(~isnan(tmp_original_distance));
        tmp_projected_distance = tmp_projected_distance(~isnan(tmp_projected_distance));
        original_distance = mean(tmp_original_distance);
        projected_distance = mean(tmp_projected_distance);
      end

      % HERE, WE MUST MENTALLY ASSERT THAT ORIGINAL_DISTANCE AND PROJECTED_DISTANCE
      % ARE SINGLE VALUES... NOT ARRAYS OF MULTIPLE VALUES. THIS IS DONE BECAUSE
      % LATER WE CAN GET BOTH ABSOLUTE VALUES AND / OR RATIO OF VALUES.

      % -------------------------------------------------------------------------
      %                                                              Finally, ...
      % -------------------------------------------------------------------------
      projected_2_original_distance_ratios(end + 1) = projected_distance / original_distance;
      original_distance_absolute_values(end + 1) = original_distance; % ignoring projected_distance... only 1 imdb passed into both anyways lol


      % TODO: naming like this: projected_2_original_distance_ratios

    end
  end

  % keyboard

  projected_2_original_distance_ratios = projected_2_original_distance_ratios(~isnan(projected_2_original_distance_ratios));
  original_distance_absolute_values = original_distance_absolute_values(~isnan(original_distance_absolute_values));
  original_distance_absolute_values_all = original_distance_absolute_values_all(~isnan(original_distance_absolute_values_all));

  average_of_all_original_distance_absolute_values = mean(original_distance_absolute_values_all); % don't actually need the / 2 because its mean... think about it!;
  % keyboard


% -------------------------------------------------------------------------
function [matrix_pdist, labels_train] = getDistanceMatrixAndLabels(imdb)
% -------------------------------------------------------------------------
  data_train = imdb.images.data(:,:,:,imdb.images.set == 1);
  labels_train = imdb.images.labels(imdb.images.set == 1);
  sample_size = size(data_train, 1) * size(data_train, 2) * size(data_train, 3);
  samples = reshape(data_train, sample_size, [])';
  matrix_pdist = squareform(pdist(samples));


% -------------------------------------------------------------------------
function distance = calculateDistance(imdb, point_1_index, point_2_index, distance_type, pdist_matrix)
% -------------------------------------------------------------------------

  point_1 = getVectorizedSampleAtIndex(imdb, point_1_index);
  point_2 = getVectorizedSampleAtIndex(imdb, point_2_index);

  if strcmp(distance_type, 'euclidean')
    % TODO: these distances are sadly not normalized :|
    % distance = pdist([point_1; point_2]);
    assert(pdist_matrix(point_1_index, point_2_index) == pdist_matrix(point_2_index, point_1_index));
    distance = pdist_matrix(point_1_index, point_2_index);
    amirs_calculated_distance = norm(point_1 - point_2);
    assert(amirs_calculated_distance - distance < 10e-3);
  elseif strcmp(distance_type, 'cosine')
    cos_theta = dot(point_1, point_2) / (norm(point_1) * norm(point_2));
    % if isnan(cos_theta)
    %   % keyboard
    %   % most likely because the denominator above is 0 ...
    %   % most likely because one of the vectors is the zero vector ...
    %   cos_theta = 1;
    % end

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
  while 1
    random_point_index = ceil(rand() * length(point_row));
    if labels(random_point_index) ~= point_class
      break
    end
  end


% -------------------------------------------------------------------------
function random_point_index = findRandomWithinClassPointIndexToPoint(point_row, point_class, labels)
% -------------------------------------------------------------------------
  while 1
    random_point_index = ceil(rand() * length(point_row));
    if labels(random_point_index) == point_class
      break
    end
  end


% -------------------------------------------------------------------------
function all_other_point_indicies = findAllOtherBetweenClassPointIndicesToPoint(original_reference_point_index, point_class, labels)
% -------------------------------------------------------------------------
  all_other_point_indicies = setdiff(find(labels ~= point_class), original_reference_point_index);


% -------------------------------------------------------------------------
function all_other_point_indicies = findAllOtherWithinClassPointIndicesToPoint(original_reference_point_index, point_class, labels)
% -------------------------------------------------------------------------
  all_other_point_indicies = setdiff(find(labels == point_class), original_reference_point_index);


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


% -------------------------------------------------------------------------
function subplotBeefAbsoluteDistance(experiments, within_between, distance_type)
% -------------------------------------------------------------------------
  color_palette = {'c', 'r', 'g', 'b', 'k'};
  legend_entries = {};
  for i = 1 : numel(experiments)
    legend_entries{i} = sprintf('%s - metric = %.4f', experiments{i}.title, experiments{i}.class_metric);
  end

  if strcmp(distance_type, 'cosine')
    x_ticks = 0:2.5:180;
    y_limits = [0 200];
  else
    x_ticks = 0:0.1:10;
    y_limits = [0 200];
  end

  title(sprintf('%s-class %s Distances', within_between, distance_type));

  hold on
  for i = 1 : numel(experiments)
    data = experiments{i}.distance_absolute_values;
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
function plotBeefAlexRatioDistance(experiments, distance_type)
% -------------------------------------------------------------------------
  color_palette = {'c', 'r', 'g', 'b', 'k'};
  legend_entries = {};
  for i = 1 : numel(experiments)
    legend_entries{i} = sprintf('%s - fd = %.4f', experiments{i}.title, experiments{i}.fisher_discriminant_ratio);
    % legend_entries{i} = experiments{i}.title;
  end

  x_ticks = 0:0.025:3.5;
  y_limits = [0 100];
  title(sprintf('Between-class to Within-class %s Distance Ratios', distance_type));

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





















































