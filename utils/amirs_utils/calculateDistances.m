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
  dataset = 'cifar-multi-class-subsampled';
  posneg_balance = 'balanced-707';
  % dataset = 'cifar-two-class-deer-truck';
  % posneg_balance = 'balanced-707';

  fh_projection_utils = projectionUtils;

  afprintf(sprintf('[INFO] Loading original imdb...\n'));
  tmp_opts.dataset = dataset;
  tmp_opts.posneg_balance = posneg_balance;
  original_imdb = loadSavedImdb(tmp_opts, 1);
  afprintf(sprintf('[INFO] done!\n'));

  original_imdb = filterImdbForSet(original_imdb, 1, 1);

  afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  projected_imdb_1 = fh_projection_utils.getDenslyProjectedImdb(original_imdb);
  afprintf(sprintf('[INFO] done!\n'));

  % afprintf(sprintf('[INFO] Applying LDA...\n'));
  % tmp_size = size(original_imdb.images.data, 1) * size(original_imdb.images.data, 2) * size(original_imdb.images.data, 3);
  % vectorized_data = reshape(original_imdb.images.data, tmp_size, [])';
  % X = vectorized_data;
  % Y = reshape(original_imdb.images.labels, [], 1);
  % W = LDA(X,Y);
  % L = [ones(length(original_imdb.images.labels),1) X] * W';
  % projected_imdb_1 = original_imdb;
  % projected_imdb_1.images.data = reshape(L', size(L, 2), 1, 1, []);
  % afprintf(sprintf('[INFO] done!\n'));


  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV1P0-ensemble-sparse-rp-no-nl';
  % larp_weight_init_sequence = getLarpWeightInitSequence(larp_weight_init_type, larp_network_arch);
  % projection_net = fh_projection_utils.getProjectionNetworkObject(dataset, larp_network_arch, larp_weight_init_sequence);
  % projected_imdb_2 = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, -1);
  % afprintf(sprintf('[INFO] done!\n'));


  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV1P0-ensemble-sparse-rp-yes-nl';
  % larp_weight_init_sequence = getLarpWeightInitSequence(larp_weight_init_type, larp_network_arch);
  % projection_net = fh_projection_utils.getProjectionNetworkObject(dataset, larp_network_arch, larp_weight_init_sequence);
  % projected_imdb_3 = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, -1);
  % afprintf(sprintf('[INFO] done!\n'));

  afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  larp_network_arch = 'larpV3P3';
  larp_weight_init_sequence = getLarpWeightInitSequence(larp_weight_init_type, larp_network_arch);
  projection_net = fh_projection_utils.getProjectionNetworkObject(dataset, larp_network_arch, larp_weight_init_sequence);
  projected_imdb_3 = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, -1);
  afprintf(sprintf('[INFO] done!\n'));

  afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV1P0-ensemble-sparse-rp-yes-nl';
  % larp_weight_init_sequence = getLarpWeightInitSequence(larp_weight_init_type, larp_network_arch);
  % tmp = load('/Volumes/Amir/matconvnet/experiment_results/temp-test-30-May-2017-15-10-53-GPU-2/test-larp-tests-30-May-2017-15-10-53-cifar-multi-class-subsampled-balanced-38-GPU-2/k=3-fold-cifar-multi-class-subsampled-30-May-2017-17-46-46-single-cnn/cnn-30-May-2017-17-46-48-cifar-multi-class-subsampled-convV1P1-RF32CH3+fcV1-RF16CH64-batch-size-100-weight-decay-0.0100-GPU-2-bpd-07/net-epoch-100.mat');

  if ispc
    datapath = 'H:\Amir\';
    path_1 = fullfile(datapath, 'some trained networks/balanced-38/v3p3/cifar/k=3-fold-cifar-multi-class-subsampled-31-May-2017-07-07-19-single-cnn/cnn-31-May-2017-07-07-20-cifar-multi-class-subsampled-convV3P3-RF32CH3+fcV1-RF4CH64-batch-size-100-weight-decay-0.0100-GPU-2-bpd-13/net-epoch-100.mat');
    path_2 = fullfile(datapath, 'Parent11.mat');
  else
    path_1 = '/Volumes/Amir/some trained networks/balanced-38/v3p3/cifar/k=3-fold-cifar-multi-class-subsampled-31-May-2017-07-07-19-single-cnn/cnn-31-May-2017-07-07-20-cifar-multi-class-subsampled-convV3P3-RF32CH3+fcV1-RF4CH64-batch-size-100-weight-decay-0.0100-GPU-2-bpd-13/net-epoch-100.mat';
    path_2 = '/Volumes/Amir/Parent11.mat';
  end

  tmp = load(path_1);
  projection_net = tmp.net;
  projected_imdb_4 = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 9);
  afprintf(sprintf('[INFO] done!\n'));
  tmp = load(path_2);
  projection_net = tmp.net;
  projected_imdb_5 = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 9);
  afprintf(sprintf('[INFO] done!\n'));
  % projected_imdb_6 = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 10);
  % afprintf(sprintf('[INFO] done!\n'));
  % projected_imdb_7 = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 12);
  % afprintf(sprintf('[INFO] done!\n'));



  % -------------------------------------------------------------------------
  %                                                                Get ratios
  % -------------------------------------------------------------------------
  point_type = 'border';
  % point_type = 'random';
  distance_type = 'euclidean';
  % distance_type = 'cosine';

  [between_class_point_ratios_1, within_class_point_ratios_1] = getPointDistanceRatios(original_imdb, projected_imdb_1, point_type, distance_type);
  % [between_class_point_ratios_2, within_class_point_ratios_2] = getPointDistanceRatios(original_imdb, projected_imdb_2, point_type, distance_type);
  [between_class_point_ratios_3, within_class_point_ratios_3] = getPointDistanceRatios(original_imdb, projected_imdb_3, point_type, distance_type);
  [between_class_point_ratios_4, within_class_point_ratios_4] = getPointDistanceRatios(original_imdb, projected_imdb_4, point_type, distance_type);
  [between_class_point_ratios_5, within_class_point_ratios_5] = getPointDistanceRatios(original_imdb, projected_imdb_5, point_type, distance_type);
  % [between_class_point_ratios_6, within_class_point_ratios_6] = getPointDistanceRatios(original_imdb, projected_imdb_6, point_type, distance_type);
  % [between_class_point_ratios_7, within_class_point_ratios_7] = getPointDistanceRatios(original_imdb, projected_imdb_7, point_type, distance_type);



  % -------------------------------------------------------------------------
  %                                                                      Plot
  % -------------------------------------------------------------------------
  figure
  % title(sprintf('%s points', point_type));

  subplot(1,2,1)
  title('Between-class Euclidean Distances')
  hold on
  histogram(between_class_point_ratios_1, 0:0.05:2, 'facecolor', 'c', 'facealpha', 0.5, 'edgecolor', 'none')
  % histogram(between_class_point_ratios_2, 0:0.05:2, 'facecolor', 'g', 'facealpha', 0.5, 'edgecolor', 'none')
  histogram(between_class_point_ratios_3, 0:0.05:2, 'facecolor', 'r', 'facealpha', 0.5, 'edgecolor', 'none')
  histogram(between_class_point_ratios_4, 0:0.05:2, 'facecolor', 'b', 'facealpha', 0.5, 'edgecolor', 'none')
  histogram(between_class_point_ratios_5, 0:0.05:2, 'facecolor', 'g', 'facealpha', 0.5, 'edgecolor', 'none')
  % histogram(between_class_point_ratios_6, 0:0.05:2, 'facecolor', 'g', 'facealpha', 0.5, 'edgecolor', 'none')
  % histogram(between_class_point_ratios_7, 0:0.05:2, 'facecolor', 'k', 'facealpha', 0.5, 'edgecolor', 'none')
  hold off
  % legend('Dense RP', 'Random Gaussian LeNet', 'Trained LeNet');
  legend('Dense RP', 'Random Gaussian LeNet', 'Trained LeNet - 38', 'Trained LeNet - ALL');

  subplot(1,2,2)
  title('Within-class Euclidean Distances')
  hold on
  histogram(within_class_point_ratios_1, 0:0.05:2, 'facecolor', 'c', 'facealpha', 0.5, 'edgecolor', 'none')
  % histogram(within_class_point_ratios_2, 0:0.05:2, 'facecolor', 'g', 'facealpha', 0.5, 'edgecolor', 'none')
  histogram(within_class_point_ratios_3, 0:0.05:2, 'facecolor', 'r', 'facealpha', 0.5, 'edgecolor', 'none')
  histogram(within_class_point_ratios_4, 0:0.05:2, 'facecolor', 'b', 'facealpha', 0.5, 'edgecolor', 'none')
  histogram(within_class_point_ratios_5, 0:0.05:2, 'facecolor', 'g', 'facealpha', 0.5, 'edgecolor', 'none')
  % histogram(within_class_point_ratios_6, 0:0.05:2, 'facecolor', 'g', 'facealpha', 0.5, 'edgecolor', 'none')
  % histogram(within_class_point_ratios_7, 0:0.05:2, 'facecolor', 'k', 'facealpha', 0.5, 'edgecolor', 'none')
  hold off
  % legend('Dense RP', 'Random Gaussian LeNet', 'Trained LeNet');
  legend('Dense RP', 'Random Gaussian LeNet', 'Trained LeNet - 38', 'Trained LeNet - ALL');

  suptitle(sprintf('%s points', point_type));

  % keyboard

  % jigar tala






% -------------------------------------------------------------------------
function [between_class_point_ratios, within_class_point_ratios] = getPointDistanceRatios(original_imdb, projected_imdb, point_type, distance_type)
% -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] Getting point distance ratios...\n'));
  % -------------------------------------------------------------------------
  %                                         Get pair-wise distances of points
  % -------------------------------------------------------------------------
  [original_pdist_matrix, original_labels_train] = getNormalizedDistanceMatrixAndLabels(original_imdb, distance_type);
  [projected_pdist_matrix, projected_labels_train] = getNormalizedDistanceMatrixAndLabels(projected_imdb, distance_type);

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


% -------------------------------------------------------------------------
function [matrix_pdist, labels_train] = getNormalizedDistanceMatrixAndLabels(imdb, distance_type)
% -------------------------------------------------------------------------
  data_train = imdb.images.data(:,:,:,imdb.images.set == 1);
  labels_train = imdb.images.labels(imdb.images.set == 1);
  sample_size = size(data_train, 1) * size(data_train, 2) * size(data_train, 3);
  samples = reshape(data_train, sample_size, [])';
  matrix_pdist = squareform(pdist(samples));
  % MAX-NORMALIZED
  % matrix_pdist = (matrix_pdist - min(matrix_pdist(:))) / (max(matrix_pdist(:)) - min(matrix_pdist(:)));
  % SUM-NORMALIZED
  matrix_pdist = matrix_pdist / (sum(matrix_pdist(:)) / 2); % remember pdist is symmetric



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
















































