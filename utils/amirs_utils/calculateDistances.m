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
  dataset = 'cifar-multi-class-subsampled'; % 'cifar';
  posneg_balance = 'balanced-38'; % 'whatever';
  fh_projection_utils = projectionUtils;


  afprintf(sprintf('[INFO] Loading original imdb...\n'));
  tmp_opts.dataset = dataset;
  tmp_opts.posneg_balance = posneg_balance;
  original_imdb = loadSavedImdb(tmp_opts, 0);
  afprintf(sprintf('[INFO] done!\n'));

  original_imdb = filterImdbForSet(original_imdb, 1, 1);


  afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  projected_imdb_1 = fh_projection_utils.getDenslyProjectedImdb(original_imdb);
  afprintf(sprintf('[INFO] done!\n'));


  afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  larp_network_arch = 'larpV1P0-ensemble-sparse-rp-no-nl';
  larp_weight_init_sequence = getLarpWeightInitSequence(larp_weight_init_type, larp_network_arch);
  projection_net = fh_projection_utils.getProjectionNetworkObject(dataset, larp_network_arch, larp_weight_init_sequence);
  projected_imdb_2 = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, -1);
  afprintf(sprintf('[INFO] done!\n'));


  afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  larp_network_arch = 'larpV1P0-ensemble-sparse-rp-yes-nl';
  larp_weight_init_sequence = getLarpWeightInitSequence(larp_weight_init_type, larp_network_arch);
  projection_net = fh_projection_utils.getProjectionNetworkObject(dataset, larp_network_arch, larp_weight_init_sequence);
  projected_imdb_3 = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, -1);
  afprintf(sprintf('[INFO] done!\n'));

  afprintf(sprintf('[INFO] Loading projected imdb...\n'));
  % larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_network_arch = 'larpV1P0-ensemble-sparse-rp-yes-nl';
  % larp_weight_init_sequence = getLarpWeightInitSequence(larp_weight_init_type, larp_network_arch);
  tmp = load('/Volumes/Amir/matconvnet/experiment_results/temp-test-30-May-2017-15-10-53-GPU-2/test-larp-tests-30-May-2017-15-10-53-cifar-multi-class-subsampled-balanced-38-GPU-2/k=3-fold-cifar-multi-class-subsampled-30-May-2017-17-46-46-single-cnn/cnn-30-May-2017-17-46-48-cifar-multi-class-subsampled-convV1P1-RF32CH3+fcV1-RF16CH64-batch-size-100-weight-decay-0.0100-GPU-2-bpd-07/net-epoch-100.mat');
  projection_net = tmp.net;
  projected_imdb_4 = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, 3);
  afprintf(sprintf('[INFO] done!\n'));



  % -------------------------------------------------------------------------
  %                                                                Get ratios
  % -------------------------------------------------------------------------
  [closest_inter_class_point_ratios_1, furthest_intra_class_point_ratios_1] = getDistanceRatios(original_imdb, projected_imdb_1);
  [closest_inter_class_point_ratios_2, furthest_intra_class_point_ratios_2] = getDistanceRatios(original_imdb, projected_imdb_2);
  [closest_inter_class_point_ratios_3, furthest_intra_class_point_ratios_3] = getDistanceRatios(original_imdb, projected_imdb_3);
  [closest_inter_class_point_ratios_4, furthest_intra_class_point_ratios_4] = getDistanceRatios(original_imdb, projected_imdb_4);



  % -------------------------------------------------------------------------
  %                                                                      Plot
  % -------------------------------------------------------------------------
  figure

  subplot(1,2,1)
  title('Inter-class Euclidean Distances')
  hold on
  histogram(closest_inter_class_point_ratios_1, 0:0.05:2, 'facecolor', 'r', 'facealpha', 0.5, 'edgecolor', 'none')
  histogram(closest_inter_class_point_ratios_2, 0:0.05:2, 'facecolor', 'g', 'facealpha', 0.5, 'edgecolor', 'none')
  histogram(closest_inter_class_point_ratios_3, 0:0.05:2, 'facecolor', 'b', 'facealpha', 0.5, 'edgecolor', 'none')
  histogram(closest_inter_class_point_ratios_4, 0:0.05:2, 'facecolor', 'c', 'facealpha', 0.5, 'edgecolor', 'none')
  hold off

  subplot(1,2,2)
  title('Intra-class Euclidean Distances')
  hold on
  histogram(furthest_intra_class_point_ratios_1, 0:0.05:2, 'facecolor', 'r', 'facealpha', 0.5, 'edgecolor', 'none')
  histogram(furthest_intra_class_point_ratios_2, 0:0.05:2, 'facecolor', 'g', 'facealpha', 0.5, 'edgecolor', 'none')
  histogram(furthest_intra_class_point_ratios_3, 0:0.05:2, 'facecolor', 'b', 'facealpha', 0.5, 'edgecolor', 'none')
  histogram(furthest_intra_class_point_ratios_4, 0:0.05:2, 'facecolor', 'c', 'facealpha', 0.5, 'edgecolor', 'none')
  hold off

  keyboard
  % figure,
  % subplot(1,2,1), histogram(closest_inter_class_point_ratios);
  % subplot(1,2,2), histogram(furthest_intra_class_point_ratios);







% -------------------------------------------------------------------------
function [closest_inter_class_point_ratios, furthest_intra_class_point_ratios] = getDistanceRatios(original_imdb, projected_imdb)
% -------------------------------------------------------------------------
  % -------------------------------------------------------------------------
  %                                         Get pair-wise distances of points
  % -------------------------------------------------------------------------
  original_data_train = original_imdb.images.data(:,:,:,original_imdb.images.set == 1);
  original_labels_train = original_imdb.images.labels(original_imdb.images.set == 1);
  original_sample_size = size(original_data_train, 1) * size(original_data_train, 2) * size(original_data_train, 3);
  original_samples = reshape(original_data_train, original_sample_size, [])';
  original_matrix_pdist = squareform(pdist(original_samples));

  projected_data_train = projected_imdb.images.data(:,:,:,projected_imdb.images.set == 1);
  projected_labels_train = projected_imdb.images.labels(projected_imdb.images.set == 1);
  projected_sample_size = size(projected_data_train, 1) * size(projected_data_train, 2) * size(projected_data_train, 3);
  projected_samples = reshape(projected_data_train, projected_sample_size, [])';
  projected_matrix_pdist = squareform(pdist(projected_samples));

  assert(size(original_matrix_pdist, 1) == size(projected_matrix_pdist, 1));
  assert(size(original_matrix_pdist, 2) == size(projected_matrix_pdist, 2));

  % -------------------------------------------------------------------------
  %                                                  Get ratios of all points
  % -------------------------------------------------------------------------
  closest_inter_class_point_ratios = [];
  furthest_intra_class_point_ratios = [];
  for point_index = 1 : size(original_matrix_pdist, 1)
    % sort distances of points from the point in question (ignoring 0, which is
    % probably the point itself and potentially other points identical to the
    % point in question).
    original_point_row = original_matrix_pdist(point_index, :);
    original_point_class = original_labels_train(point_index);
    original_closest_inter_class_point = findClosestInterClassPointToPoint(original_point_row, original_point_class, original_labels_train);
    original_furthest_intra_class_point = findFurthestIntraClassPointToPoint(original_point_row, original_point_class, original_labels_train);

    projected_point_row = projected_matrix_pdist(point_index, :);
    projected_point_class = projected_labels_train(point_index);
    projected_closest_inter_class_point = findClosestInterClassPointToPoint(projected_point_row, projected_point_class, projected_labels_train);
    projected_furthest_intra_class_point = findFurthestIntraClassPointToPoint(projected_point_row, projected_point_class, projected_labels_train);

    closest_inter_class_point_ratios(end + 1) = projected_closest_inter_class_point / original_closest_inter_class_point;
    furthest_intra_class_point_ratios(end + 1) = projected_furthest_intra_class_point / original_furthest_intra_class_point;
  end



% -------------------------------------------------------------------------
function distance = findClosestInterClassPointToPoint(point_row, point_class, labels)
% -------------------------------------------------------------------------
  sorted_point_row = sort(point_row(labels ~= point_class));
  distance = sorted_point_row(1);


% -------------------------------------------------------------------------
function distance = findFurthestIntraClassPointToPoint(point_row, point_class, labels)
% -------------------------------------------------------------------------
  sorted_point_row = sort(point_row(labels == point_class));
  distance = sorted_point_row(end);


















































