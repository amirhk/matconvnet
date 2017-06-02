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
  posneg_balance = 'balanced-266';
  % dataset = 'cifar-two-class-deer-truck';
  % posneg_balance = 'balanced-707';

  fh_projection_utils = projectionUtils;

  afprintf(sprintf('[INFO] Loading original imdb...\n'));
  tmp_opts.dataset = dataset;
  tmp_opts.posneg_balance = posneg_balance;
  original_imdb = loadSavedImdb(tmp_opts, 1);
  afprintf(sprintf('[INFO] done!\n'));

  original_imdb = filterImdbForSet(original_imdb, 1, 1);

  % afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % projected_imdb_1 = fh_projection_utils.getDenslyProjectedImdb(original_imdb);
  % afprintf(sprintf('[INFO] done!\n'));

  afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  larp_network_arch = 'larpV3P3';
  larp_weight_init_sequence = getLarpWeightInitSequence(larp_weight_init_type, larp_network_arch);
  projection_net = fh_projection_utils.getProjectionNetworkObject(dataset, larp_network_arch, larp_weight_init_sequence);
  projected_imdb_1 = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, -1);
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
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV1P0-ensemble-sparse-rp-yes-nl';
  % larp_weight_init_sequence = getLarpWeightInitSequence(larp_weight_init_type, larp_network_arch);
  % tmp = load('/Volumes/Amir/matconvnet/experiment_results/temp-test-30-May-2017-15-10-53-GPU-2/test-larp-tests-30-May-2017-15-10-53-cifar-multi-class-subsampled-balanced-38-GPU-2/k=3-fold-cifar-multi-class-subsampled-30-May-2017-17-46-46-single-cnn/cnn-30-May-2017-17-46-48-cifar-multi-class-subsampled-convV1P1-RF32CH3+fcV1-RF16CH64-batch-size-100-weight-decay-0.0100-GPU-2-bpd-07/net-epoch-100.mat');
  tmp = load('/Volumes/Amir/Parent11.mat');
  projection_net = tmp.net;
  % projected_imdb_4 = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 1);
  % afprintf(sprintf('[INFO] done!\n'));
  projected_imdb_5 = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 9);
  afprintf(sprintf('[INFO] done!\n'));
  % projected_imdb_6 = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 10);
  % afprintf(sprintf('[INFO] done!\n'));
  % projected_imdb_7 = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 12);
  % afprintf(sprintf('[INFO] done!\n'));



  % -------------------------------------------------------------------------
  %                                                                Get ratios
  % -------------------------------------------------------------------------
  % distance_type = 'euclidean';
  distance_type = 'cosine';
  % point_type = 'border';
  % point_type = 'random';
  [between_class_point_ratios_1, within_class_point_ratios_1] = getRandomPointDistanceRatios(original_imdb, projected_imdb_1, distance_type);
  % [between_class_point_ratios_2, within_class_point_ratios_2] = getRandomPointDistanceRatios(original_imdb, projected_imdb_2, distance_type);
  % [between_class_point_ratios_3, within_class_point_ratios_3] = getRandomPointDistanceRatios(original_imdb, projected_imdb_3, distance_type);
  % [between_class_point_ratios_4, within_class_point_ratios_4] = getRandomPointDistanceRatios(original_imdb, projected_imdb_4, distance_type);
  [between_class_point_ratios_5, within_class_point_ratios_5] = getRandomPointDistanceRatios(original_imdb, projected_imdb_5, distance_type);
  % [between_class_point_ratios_6, within_class_point_ratios_6] = getRandomPointDistanceRatios(original_imdb, projected_imdb_6, distance_type);
  % [between_class_point_ratios_7, within_class_point_ratios_7] = getRandomPointDistanceRatios(original_imdb, projected_imdb_7, distance_type);



  % -------------------------------------------------------------------------
  %                                                                      Plot
  % -------------------------------------------------------------------------
  figure

  subplot(1,2,1)
  title('Between-class Euclidean Distances')
  hold on
  histogram(between_class_point_ratios_1, 0:0.05:2, 'facecolor', 'r', 'facealpha', 0.5, 'edgecolor', 'none')
  % histogram(between_class_point_ratios_2, 0:0.05:2, 'facecolor', 'g', 'facealpha', 0.5, 'edgecolor', 'none')
  % histogram(between_class_point_ratios_3, 0:0.05:2, 'facecolor', 'b', 'facealpha', 0.5, 'edgecolor', 'none')
  % histogram(between_class_point_ratios_4, 0:0.05:2, 'facecolor', 'c', 'facealpha', 0.5, 'edgecolor', 'none')
  histogram(between_class_point_ratios_5, 0:0.05:2, 'facecolor', 'b', 'facealpha', 0.5, 'edgecolor', 'none')
  % histogram(between_class_point_ratios_6, 0:0.05:2, 'facecolor', 'g', 'facealpha', 0.5, 'edgecolor', 'none')
  % histogram(between_class_point_ratios_7, 0:0.05:2, 'facecolor', 'k', 'facealpha', 0.5, 'edgecolor', 'none')
  hold off
  legend('Random Gaussian LeNet', 'Trained LeNet');

  subplot(1,2,2)
  title('Within-class Euclidean Distances')
  hold on
  histogram(within_class_point_ratios_1, 0:0.05:2, 'facecolor', 'r', 'facealpha', 0.5, 'edgecolor', 'none')
  % histogram(within_class_point_ratios_2, 0:0.05:2, 'facecolor', 'g', 'facealpha', 0.5, 'edgecolor', 'none')
  % histogram(within_class_point_ratios_3, 0:0.05:2, 'facecolor', 'b', 'facealpha', 0.5, 'edgecolor', 'none')
  % histogram(within_class_point_ratios_4, 0:0.05:2, 'facecolor', 'c', 'facealpha', 0.5, 'edgecolor', 'none')
  histogram(within_class_point_ratios_5, 0:0.05:2, 'facecolor', 'b', 'facealpha', 0.5, 'edgecolor', 'none')
  % histogram(within_class_point_ratios_6, 0:0.05:2, 'facecolor', 'g', 'facealpha', 0.5, 'edgecolor', 'none')
  % histogram(within_class_point_ratios_7, 0:0.05:2, 'facecolor', 'k', 'facealpha', 0.5, 'edgecolor', 'none')
  hold off
  legend('Random Gaussian LeNet', 'Trained LeNet');

  keyboard





% -------------------------------------------------------------------------
function [closest_between_class_point_ratios, furthest_within_class_point_ratios] = getBorderPointDistanceRatios(original_imdb, projected_imdb, distance_type)
% -------------------------------------------------------------------------
  % -------------------------------------------------------------------------
  %                                         Get pair-wise distances of points
  % -------------------------------------------------------------------------
  [original_matrix_pdist, original_labels_train] = getDistanceMatrixAndLabels(original_imdb, distance_type);
  [projected_matrix_pdist, projected_labels_train] = getDistanceMatrixAndLabels(projected_imdb, distance_type);

  assert(size(original_matrix_pdist, 1) == size(projected_matrix_pdist, 1));
  assert(size(original_matrix_pdist, 2) == size(projected_matrix_pdist, 2));

  % -------------------------------------------------------------------------
  %                                                  Get ratios of all points
  % -------------------------------------------------------------------------
  closest_between_class_point_ratios = [];
  furthest_within_class_point_ratios = [];
  for point_index = 1 : size(original_matrix_pdist, 1)

    % -------------------------------------------------------------------------
    %                                                         Get point indices
    % -------------------------------------------------------------------------
    original_point_row = original_matrix_pdist(point_index, :);
    original_point_class = original_labels_train(point_index);
    original_closest_between_class_point_index = findClosestBetweenClassPointToPoint(original_point_row, original_point_class, original_labels_train);
    original_furthest_within_class_point_index = findFurthestWithinClassPointToPoint(original_point_row, original_point_class, original_labels_train);

    projected_point_row = projected_matrix_pdist(point_index, :);
    projected_point_class = projected_labels_train(point_index);
    projected_closest_between_class_point_index = findClosestBetweenClassPointToPoint(projected_point_row, projected_point_class, projected_labels_train);
    projected_furthest_within_class_point_index = findFurthestWithinClassPointToPoint(projected_point_row, projected_point_class, projected_labels_train);


    % -------------------------------------------------------------------------
    %                                                     Get points themselves
    % -------------------------------------------------------------------------
    original_reference_point = getVectorizedSampleAtIndex(original_imdb, point_index);
    original_closest_between_class_point = getVectorizedSampleAtIndex(original_imdb, original_closest_between_class_point_index);
    original_furthest_within_class_point = getVectorizedSampleAtIndex(original_imdb, original_furthest_within_class_point_index);

    projected_reference_point = getVectorizedSampleAtIndex(projected_imdb, point_index);
    projected_closest_between_class_point = getVectorizedSampleAtIndex(projected_imdb, projected_closest_between_class_point_index);
    projected_furthest_within_class_point = getVectorizedSampleAtIndex(projected_imdb, projected_furthest_within_class_point_index);


    % -------------------------------------------------------------------------
    %                                                        Calculate distance
    % -------------------------------------------------------------------------
    original_closest_between_class_distance = getDistance(original_reference_point, original_closest_between_class_point, distance_type);
    original_furthest_within_class_distance = getDistance(original_reference_point, original_furthest_within_class_point, distance_type);
    projected_closest_between_class_distance = getDistance(projected_reference_point, projected_closest_between_class_point, distance_type);
    projected_furthest_within_class_distance = getDistance(projected_reference_point, projected_furthest_within_class_point, distance_type);


    % -------------------------------------------------------------------------
    %                                                 Finally, calculate ratios
    % -------------------------------------------------------------------------
    % TODO: these distances are sadly not normalized :|
    closest_between_class_point_ratios(end + 1) = projected_closest_between_class_distance / original_closest_between_class_distance;
    furthest_within_class_point_ratios(end + 1) = projected_furthest_within_class_distance / original_furthest_within_class_distance;

  end


% -------------------------------------------------------------------------
function [random_between_class_point_ratios, random_within_class_point_ratios] = getRandomPointDistanceRatios(original_imdb, projected_imdb, distance_type)
% -------------------------------------------------------------------------
  % -------------------------------------------------------------------------
  %                                         Get pair-wise distances of points
  % -------------------------------------------------------------------------
  [original_matrix_pdist, original_labels_train] = getDistanceMatrixAndLabels(original_imdb, distance_type);
  [projected_matrix_pdist, projected_labels_train] = getDistanceMatrixAndLabels(projected_imdb, distance_type);

  assert(size(original_matrix_pdist, 1) == size(projected_matrix_pdist, 1));
  assert(size(original_matrix_pdist, 2) == size(projected_matrix_pdist, 2));



  % -------------------------------------------------------------------------
  %                                                  Get ratios of all points
  % -------------------------------------------------------------------------
  random_between_class_point_ratios = [];
  random_within_class_point_ratios = [];
  for i = 1:10
    for point_index = 1 : size(original_matrix_pdist, 1)

      % -------------------------------------------------------------------------
      %                                                         Get point indices
      % -------------------------------------------------------------------------
      original_point_row = original_matrix_pdist(point_index, :);
      original_point_class = original_labels_train(point_index);
      original_closest_between_class_point_index = findRandomBetweenClassPointToPoint(original_point_row, original_point_class, original_labels_train);
      original_furthest_within_class_point_index = findRandomWithinClassPointToPoint(original_point_row, original_point_class, original_labels_train);

      projected_point_row = projected_matrix_pdist(point_index, :);
      projected_point_class = projected_labels_train(point_index);
      projected_closest_between_class_point_index = original_closest_between_class_point_index;
      projected_furthest_within_class_point_index = original_furthest_within_class_point_index;


      % -------------------------------------------------------------------------
      %                                                     Get points themselves
      % -------------------------------------------------------------------------
      original_reference_point = getVectorizedSampleAtIndex(original_imdb, point_index);
      original_closest_between_class_point = getVectorizedSampleAtIndex(original_imdb, original_closest_between_class_point_index);
      original_furthest_within_class_point = getVectorizedSampleAtIndex(original_imdb, original_furthest_within_class_point_index);

      projected_reference_point = getVectorizedSampleAtIndex(projected_imdb, point_index);
      projected_closest_between_class_point = getVectorizedSampleAtIndex(projected_imdb, projected_closest_between_class_point_index);
      projected_furthest_within_class_point = getVectorizedSampleAtIndex(projected_imdb, projected_furthest_within_class_point_index);


      % -------------------------------------------------------------------------
      %                                                        Calculate distance
      % -------------------------------------------------------------------------
      original_closest_between_class_distance = getDistance(original_reference_point, original_closest_between_class_point, distance_type);
      original_furthest_within_class_distance = getDistance(original_reference_point, original_furthest_within_class_point, distance_type);
      projected_closest_between_class_distance = getDistance(projected_reference_point, projected_closest_between_class_point, distance_type);
      projected_furthest_within_class_distance = getDistance(projected_reference_point, projected_furthest_within_class_point, distance_type);


      % -------------------------------------------------------------------------
      %                                                 Finally, calculate ratios
      % -------------------------------------------------------------------------
      random_between_class_point_ratios(end + 1) = projected_closest_between_class_distance / original_closest_between_class_distance;
      random_within_class_point_ratios(end + 1) = projected_furthest_within_class_distance / original_furthest_within_class_distance;
    end
  end



% -------------------------------------------------------------------------
function distance = getDistance(point_1, point_2, distance_type)
% -------------------------------------------------------------------------
  if strcmp(distance_type, 'euclidean')
    distance = pdist([point_1; point_2]);
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
function [matrix_pdist, labels_train] = getDistanceMatrixAndLabels(imdb, distance_type)
% -------------------------------------------------------------------------
  data_train = imdb.images.data(:,:,:,imdb.images.set == 1);
  labels_train = imdb.images.labels(imdb.images.set == 1);
  sample_size = size(data_train, 1) * size(data_train, 2) * size(data_train, 3);
  samples = reshape(data_train, sample_size, [])';
  matrix_pdist = squareform(pdist(samples));
  matrix_pdist = (matrix_pdist - min(matrix_pdist(:))) / (max(matrix_pdist(:)) - min(matrix_pdist(:)));


% -------------------------------------------------------------------------
function point_index = findClosestBetweenClassPointToPoint(point_row, point_class, labels)
% -------------------------------------------------------------------------
  [sorted_point_row, indices] = sort(point_row(labels ~= point_class));
  point_index = indices(1);


% -------------------------------------------------------------------------
function point_index = findFurthestWithinClassPointToPoint(point_row, point_class, labels)
% -------------------------------------------------------------------------
  [sorted_point_row, indices] = sort(point_row(labels == point_class));
  point_index = indices(end);


% -------------------------------------------------------------------------
function random_point_index = findRandomBetweenClassPointToPoint(point_row, point_class, labels)
% -------------------------------------------------------------------------
  between_class_row = point_row(labels ~= point_class);
  random_point_index = ceil(rand() * length(between_class_row));


% -------------------------------------------------------------------------
function random_point_index = findRandomWithinClassPointToPoint(point_row, point_class, labels)
% -------------------------------------------------------------------------
  within_class_row = point_row(labels == point_class);
  random_point_index = ceil(rand() * length(within_class_row));

















































