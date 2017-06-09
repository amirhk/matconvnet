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
  % posneg_balance = 'balanced-100';
  dataset = 'cifar-two-class-deer-truck';
  posneg_balance = 'balanced-38';

  fh_projection_utils = projectionUtils;
  experiments = {};

  afprintf(sprintf('[INFO] Loading original imdb...\n'));
  tmp_opts.dataset = dataset;
  tmp_opts.posneg_balance = posneg_balance;
  original_imdb = loadSavedImdb(tmp_opts, 1);
  original_imdb = filterImdbForSet(original_imdb, 1, 1);
  afprintf(sprintf('[INFO] done!\n'));


  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projected_imdb = fh_projection_utils.getDenslyProjectedImdb(original_imdb);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Dense Random Projection Matrix';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Applying LDA...\n'));
  % tmp_size = size(original_imdb.images.data, 1) * size(original_imdb.images.data, 2) * size(original_imdb.images.data, 3);
  % vectorized_data = reshape(original_imdb.images.data, tmp_size, [])';
  % X = vectorized_data;
  % Y = reshape(original_imdb.images.labels, [], 1);
  % W = LDA(X,Y);
  % L = [ones(length(original_imdb.images.labels),1) X] * W';
  % projected_imdb = original_imdb;
  % projected_imdb.images.data = reshape(L', size(L, 2), 1, 1, []);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'LDA';
  % afprintf(sprintf('[INFO] done!\n'));


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV1P0RL0-ensemble-sparse-rp';
  % projected_imdb = getProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Random Gaussian V1P0 w/o ReLU';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV1P0RL1-ensemble-sparse-rp';
  % projected_imdb = getProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Random Gaussian V1P0 w ReLU';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV3P0RL0';
  % projected_imdb = getProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Random Gaussian V3P0 w/o ReLU';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV3P0RL3';
  % projected_imdb = getProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Random Gaussian V3P0 w/ ReLU';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV3P3RL0';
  % projected_imdb = getProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Random Gaussian V3P3 (LeNet) w/o ReLU';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV3P3RL3';
  % projected_imdb = getProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Random Gaussian V3P3 (LeNet) w/ ReLU';
  % afprintf(sprintf('[INFO] done!\n'));


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV1P0RL0', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 1);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained V1P0 w/o ReLU - trained on 38';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV1P0RL1', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 2);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained V1P0 w ReLU - trained on 38';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P0RL0', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 3);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained V3P0 w/o ReLU - trained on 38';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P0RL3', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 6);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained V3P0 w/ ReLU - trained on 38';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P3RL0', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 6);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained V3P3 (LeNet) w/o ReLU - trained on 38';
  % afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P3RL3', 'cifar', 'balanced-38');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 9);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained V3P3 (LeNet) w/ ReLU - trained on 38';
  % afprintf(sprintf('[INFO] done!\n'));


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % tmp = load(path_2);
  % projection_net = loadTrainedNet('larpV0P0RL0+convV3P3RL3', 'cifar', 'whatever');
  % projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 9);
  % experiments{end+1}.imdb = projected_imdb;
  % experiments{end}.title = 'Trained V3P3 (LeNet) - trained on ALL';
  % afprintf(sprintf('[INFO] done!\n'));


  if true
    % -------------------------------------------------------------------------
    %                                                                   Run KMD
    % -------------------------------------------------------------------------
    for i = 1 : numel(experiments)
      [experiments{i}.H, experiments{i}.info] = runKmdOnImdb(experiments{i}.imdb);
    end

    for i = 1 : numel(experiments)
      afprintf(sprintf( ...
        '[INFO] Results for `%s`: \t\t val = %.6f, bound = %.6f\n\n', ...
        experiments{i}.title, ...
        experiments{i}.info.mmd.val, ...
        experiments{i}.info.mmd.bound));
    end
  else
    % -------------------------------------------------------------------------
    %                                                                Get ratios
    % -------------------------------------------------------------------------
    point_type = 'border';
    % point_type = 'random';
    distance_type = 'euclidean';
    % distance_type = 'cosine';

    for i = 1 : numel(experiments)
      projected_imdb = experiments{i}.imdb;
      [between_class_point_ratios, within_class_point_ratios] = getPointDistanceRatios(original_imdb, projected_imdb, point_type, distance_type);
      experiments{i}.between_class_point_ratios = between_class_point_ratios;
      experiments{i}.within_class_point_ratios = within_class_point_ratios;
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
    title('Between-class Euclidean Distances')
    hold on
    for i = 1 : numel(experiments)
      histogram( ...
        experiments{i}.between_class_point_ratios, ...
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
    title('Within-class Euclidean Distances')
    hold on
    for i = 1 : numel(experiments)
      histogram( ...
        experiments{i}.within_class_point_ratios, ...
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

    % keyboard

    % jigar tala

  end




% -------------------------------------------------------------------------
function [between_class_point_ratios, within_class_point_ratios] = getPointDistanceRatios(original_imdb, projected_imdb, point_type, distance_type)
% -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] Getting point distance ratios...\n'));
  tic
  % -------------------------------------------------------------------------
  %                                         Get pair-wise distances of points
  % -------------------------------------------------------------------------
  % [original_pdist_matrix, original_labels_train] = getNormalizedDistanceMatrixAndLabels(original_imdb, distance_type);
  % [projected_pdist_matrix, projected_labels_train] = getNormalizedDistanceMatrixAndLabels(projected_imdb, distance_type);

  [original_pdist_matrix, original_labels_train] = getDistanceMatrixAndLabels(original_imdb, distance_type);
  [projected_pdist_matrix, projected_labels_train] = getDistanceMatrixAndLabels(projected_imdb, distance_type);

  max_o = max(original_pdist_matrix(:));
  min_o = min(original_pdist_matrix(:));
  sum_o = sum(original_pdist_matrix(:)) / 2;
  max_p = max(projected_pdist_matrix(:));
  min_p = min(projected_pdist_matrix(:));
  sum_p = sum(projected_pdist_matrix(:)) / 2;

  % % MAX-NORMALIZED
  % original_pdist_matrix = (original_pdist_matrix - min_o) / (max_o - min_o);
  % projected_pdist_matrix = (projected_pdist_matrix - min_p) / (max_p - min_p);

  % SUM-NORMALIZED
  original_pdist_matrix = original_pdist_matrix / sum_o; % remember pdist is symmetric
  projected_pdist_matrix = projected_pdist_matrix / sum_p; % remember pdist is symmetric

  % % EMAIL MAX-NORMALIZE (NORMALIZE PROJECTED SPACE INTO DYNAMIC RANGE OF INPUT SPACE)
  % original_pdist_matrix = original_pdist_matrix;
  % projected_pdist_matrix = (projected_pdist_matrix - min_p) / (max_p - min_p) * (max_o - min_o) + min_o;

  % % EMAIL SUM-NORMALIZED
  % original_pdist_matrix = original_pdist_matrix / sum_o; % remember pdist is symmetric
  % projected_pdist_matrix = projected_pdist_matrix / sum_p * sum_o; % remember pdist is symmetric

  assert(size(original_pdist_matrix, 1) == size(projected_pdist_matrix, 1));
  assert(size(original_pdist_matrix, 2) == size(projected_pdist_matrix, 2));

  % -------------------------------------------------------------------------
  %                                                  Get ratios of all points
  % -------------------------------------------------------------------------
  between_class_point_ratios = [];
  within_class_point_ratios = [];
  switch point_type
    case 'border'
      repeat_count = 1;
    case 'random'
      repeat_count = 10;
  end
  for i = 1 : repeat_count
    for point_index = 1 : size(original_pdist_matrix, 1)

      % -------------------------------------------------------------------------
      %                                                         Get point indices
      % -------------------------------------------------------------------------
      if strcmp(point_type, 'border')

        original_point_row = original_pdist_matrix(point_index, :);
        original_point_class = original_labels_train(point_index);
        original_closest_between_class_point_index = findClosestBetweenClassPointToPoint(original_point_row, original_point_class, original_labels_train);
        original_furthest_within_class_point_index = findFurthestWithinClassPointToPoint(original_point_row, original_point_class, original_labels_train);

        projected_point_row = projected_pdist_matrix(point_index, :);
        projected_point_class = projected_labels_train(point_index);
        projected_closest_between_class_point_index = findClosestBetweenClassPointToPoint(projected_point_row, projected_point_class, projected_labels_train);
        projected_furthest_within_class_point_index = findFurthestWithinClassPointToPoint(projected_point_row, projected_point_class, projected_labels_train);

      elseif strcmp(point_type, 'random')

        original_point_row = original_pdist_matrix(point_index, :);
        original_point_class = original_labels_train(point_index);
        original_closest_between_class_point_index = findRandomBetweenClassPointToPoint(original_point_row, original_point_class, original_labels_train);
        original_furthest_within_class_point_index = findRandomWithinClassPointToPoint(original_point_row, original_point_class, original_labels_train);

        projected_point_row = projected_pdist_matrix(point_index, :);
        projected_point_class = projected_labels_train(point_index);
        projected_closest_between_class_point_index = original_closest_between_class_point_index;
        projected_furthest_within_class_point_index = original_furthest_within_class_point_index;

      end

      % -------------------------------------------------------------------------
      %                                                     Get points themselves
      % -------------------------------------------------------------------------
      original_reference_point_index = point_index;
      original_reference_point = getVectorizedSampleAtIndex(original_imdb, original_reference_point_index);
      original_closest_between_class_point = getVectorizedSampleAtIndex(original_imdb, original_closest_between_class_point_index);
      original_furthest_within_class_point = getVectorizedSampleAtIndex(original_imdb, original_furthest_within_class_point_index);

      projected_reference_point_index = point_index;
      projected_reference_point = getVectorizedSampleAtIndex(projected_imdb, projected_reference_point_index);
      projected_closest_between_class_point = getVectorizedSampleAtIndex(projected_imdb, projected_closest_between_class_point_index);
      projected_furthest_within_class_point = getVectorizedSampleAtIndex(projected_imdb, projected_furthest_within_class_point_index);

      % -------------------------------------------------------------------------
      %                                                        Calculate distance
      % -------------------------------------------------------------------------
      % original_closest_between_class_distance = calculateDistance(original_reference_point, original_closest_between_class_point, distance_type, original_pdist_matrix);
      % original_furthest_within_class_distance = calculateDistance(original_reference_point, original_furthest_within_class_point, distance_type, original_pdist_matrix);
      % projected_closest_between_class_distance = calculateDistance(projected_reference_point, projected_closest_between_class_point, distance_type, projected_pdist_matrix);
      % projected_furthest_within_class_distance = calculateDistance(projected_reference_point, projected_furthest_within_class_point, distance_type, projected_pdist_matrix);
      % original_closest_between_class_distance = calculateDistance(original_reference_point, original_closest_between_class_point, distance_type);
      % original_furthest_within_class_distance = calculateDistance(original_reference_point, original_furthest_within_class_point, distance_type);
      % projected_closest_between_class_distance = calculateDistance(projected_reference_point, projected_closest_between_class_point, distance_type);
      % projected_furthest_within_class_distance = calculateDistance(projected_reference_point, projected_furthest_within_class_point, distance_type);
      original_closest_between_class_distance = calculateDistance(original_reference_point, original_reference_point_index, original_closest_between_class_point, original_closest_between_class_point_index, distance_type, original_pdist_matrix);
      original_furthest_within_class_distance = calculateDistance(original_reference_point, original_reference_point_index, original_furthest_within_class_point, original_furthest_within_class_point_index, distance_type, original_pdist_matrix);
      projected_closest_between_class_distance = calculateDistance(projected_reference_point, projected_reference_point_index, projected_closest_between_class_point, projected_closest_between_class_point_index, distance_type, projected_pdist_matrix);
      projected_furthest_within_class_distance = calculateDistance(projected_reference_point, projected_reference_point_index, projected_furthest_within_class_point, projected_furthest_within_class_point_index, distance_type, projected_pdist_matrix);

      % -------------------------------------------------------------------------
      %                                                 Finally, calculate ratios
      % -------------------------------------------------------------------------
      between_class_point_ratios(end + 1) = projected_closest_between_class_distance / original_closest_between_class_distance;
      within_class_point_ratios(end + 1) = projected_furthest_within_class_distance / original_furthest_within_class_distance;

    end
  end
  afprintf(sprintf('[INFO] done!\n'));
  toc


% % -------------------------------------------------------------------------
% function [matrix_pdist, labels_train] = getNormalizedDistanceMatrixAndLabels(imdb, distance_type)
% % -------------------------------------------------------------------------
%   data_train = imdb.images.data(:,:,:,imdb.images.set == 1);
%   labels_train = imdb.images.labels(imdb.images.set == 1);
%   sample_size = size(data_train, 1) * size(data_train, 2) * size(data_train, 3);
%   samples = reshape(data_train, sample_size, [])';
%   matrix_pdist = squareform(pdist(samples));
%   % MAX-NORMALIZED
%   % matrix_pdist = (matrix_pdist - min(matrix_pdist(:))) / (max(matrix_pdist(:)) - min(matrix_pdist(:)));
%   % SUM-NORMALIZED
%   matrix_pdist = matrix_pdist / (sum(matrix_pdist(:)) / 2); % remember pdist is symmetric


% -------------------------------------------------------------------------
function [matrix_pdist, labels_train] = getDistanceMatrixAndLabels(imdb, distance_type)
% -------------------------------------------------------------------------
  data_train = imdb.images.data(:,:,:,imdb.images.set == 1);
  labels_train = imdb.images.labels(imdb.images.set == 1);
  sample_size = size(data_train, 1) * size(data_train, 2) * size(data_train, 3);
  samples = reshape(data_train, sample_size, [])';
  matrix_pdist = squareform(pdist(samples));
  % MAX-NORMALIZED
  % matrix_pdist = (matrix_pdist - min(matrix_pdist(:))) / (max(matrix_pdist(:)) - min(matrix_pdist(:)));
  % SUM-NORMALIZED
  % matrix_pdist = matrix_pdist / (sum(matrix_pdist(:)) / 2); % remember pdist is symmetric


% -------------------------------------------------------------------------
% function distance = calculateDistance(point_1, point_2, distance_type, pdist_matrix)
  % function distance = calculateDistance(point_1, point_2, distance_type)
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
function vectorized_sample = getVectorizedSampleAtIndex(imdb, index)
% -------------------------------------------------------------------------
  vectorized_sample = vectorizeSample(imdb.images.data(:,:,:,index));


% -------------------------------------------------------------------------
function vectorized_sample = vectorizeSample(sample)
% -------------------------------------------------------------------------
  sample_size = size(sample, 1) * size(sample, 2) * size(sample, 3);
  vectorized_sample = reshape(sample, sample_size, [])';


% -------------------------------------------------------------------------
function point_index = findClosestBetweenClassPointToPoint(point_row, point_class, labels)
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
function point_index = findFurthestWithinClassPointToPoint(point_row, point_class, labels)
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
function random_point_index = findRandomBetweenClassPointToPoint(point_row, point_class, labels)
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
function random_point_index = findRandomWithinClassPointToPoint(point_row, point_class, labels)
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
function [H, info] = runKmdOnImdb(imdb)
% -------------------------------------------------------------------------
  data_train = imdb.images.data(:,:,:,imdb.images.set == 1);
  labels_train = imdb.images.labels(imdb.images.set == 1);
  sample_size = size(data_train, 1) * size(data_train, 2) * size(data_train, 3);
  samples = reshape(data_train, sample_size, [])';

  X = samples;
  labels = (-1).^labels_train';
  [H,info] = kmd(X,labels);


% -------------------------------------------------------------------------
function projected_imdb = getProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, projection_depth)
% -------------------------------------------------------------------------
  fh_projection_utils = projectionUtils;
  larp_weight_init_sequence = getLarpWeightInitSequence(larp_weight_init_type, larp_network_arch);
  projection_net = fh_projection_utils.getProjectionNetworkObject(dataset, larp_network_arch, larp_weight_init_sequence);
  projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, projection_depth);









































