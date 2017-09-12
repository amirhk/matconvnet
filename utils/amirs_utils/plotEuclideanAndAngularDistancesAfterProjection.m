% -------------------------------------------------------------------------
function plotEuclideanAndAngularDistancesAfterProjection()
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


  numbers_of_samples = 1000;
  sample_dim = 25;

  original_samples = randn(numbers_of_samples, sample_dim);

  original_pdist_angular_squareform = squareform(acosd(1 - pdist(original_samples, 'cosine')));
  original_pdist_euclidean_squareform = squareform(pdist(original_samples, 'euclidean'));


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %                                 SPLIT SAMPLE PAIRS BASED ON ANGULAR THRESHOLD
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  threshold = acosd(1/pi);

  sample_pairs_with_original_angle_less_than_threshold = {};
  sample_pairs_with_original_angle_more_than_threshold = {};

  for i = 1 : size(original_pdist_angular_squareform,1)
    for j = i : size(original_pdist_angular_squareform,1) % upper right triangle
      if original_pdist_angular_squareform(j,i) < threshold
        sample_pairs_with_original_angle_less_than_threshold{end+1} = [j,i];
      else
        sample_pairs_with_original_angle_more_than_threshold{end+1} = [j,i];
      end
    end
  end




  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %                                  GET ORIGINAL ANGULAR AND EUCLIDEAN DISTANCES
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  [original_angular_distances, original_euclidean_distances] = tmpFunction( ...
    original_pdist_angular_squareform, ...
    original_pdist_euclidean_squareform, ...
    sample_pairs_with_original_angle_less_than_threshold, ...
    sample_pairs_with_original_angle_more_than_threshold);









  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %                                                                LINEAR PROJECT
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  linear_rp_samples = original_samples * randn(sample_dim, sample_dim) / sqrt(sample_dim);

  linear_rp_pdist_angular_squareform = squareform(acosd(1 - pdist(linear_rp_samples, 'cosine')));
  linear_rp_pdist_euclidean_squareform = squareform(pdist(linear_rp_samples, 'euclidean'));



  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %                                 GET PROJECTED ANGULAR AND EUCLIDEAN DISTANCES
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  [linear_rp_angular_distances, linear_rp_euclidean_distances] = tmpFunction( ...
    linear_rp_pdist_angular_squareform, ...
    linear_rp_pdist_euclidean_squareform, ...
    sample_pairs_with_original_angle_less_than_threshold, ...
    sample_pairs_with_original_angle_more_than_threshold);














  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %                                                             RECTIFIED PROJECT
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  nonlinear_rectified_rp_samples = original_samples * randn(sample_dim, sample_dim) / sqrt(sample_dim);
  nonlinear_rectified_rp_samples(nonlinear_rectified_rp_samples < 0) = 0;

  nonlinear_rectified_rp_pdist_angular_squareform = squareform(acosd(1 - pdist(nonlinear_rectified_rp_samples, 'cosine')));
  nonlinear_rectified_rp_pdist_euclidean_squareform = squareform(pdist(nonlinear_rectified_rp_samples, 'euclidean'));



  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %                                 GET PROJECTED ANGULAR AND EUCLIDEAN DISTANCES
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  [nonlinear_rectified_rp_angular_distances, nonlinear_rectified_rp_euclidean_distances] = tmpFunction( ...
    nonlinear_rectified_rp_pdist_angular_squareform, ...
    nonlinear_rectified_rp_pdist_euclidean_squareform, ...
    sample_pairs_with_original_angle_less_than_threshold, ...
    sample_pairs_with_original_angle_more_than_threshold);















  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %                                                                 MAX 2 PROJECT
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  max_count = 2;
  nonlinear_max_2_rp_samples = original_samples * randn(sample_dim, max_count * sample_dim) / sqrt(sample_dim);
  % nonlinear_max_2_rp_samples(nonlinear_max_2_rp_samples < 0) = 0;
  tmp_nonlinear_max_2_rp_samples = zeros(size(original_samples, 1), size(original_samples, 2));
  for j = 1 : size(tmp_nonlinear_max_2_rp_samples, 1)
    for i = 1 : size(tmp_nonlinear_max_2_rp_samples, 2)
      % nonlinear_max_2_rp_samples(j, (i-1)*max_count+1:(i-1)*max_count+1+max_count-1)
      % keyboard
      tmp_nonlinear_max_2_rp_samples(j,i) = max(nonlinear_max_2_rp_samples(j, (i-1)*max_count+1:(i-1)*max_count+1+max_count-1));
    end
  end

  nonlinear_max_2_rp_pdist_angular_squareform = squareform(acosd(1 - pdist(nonlinear_max_2_rp_samples, 'cosine')));
  nonlinear_max_2_rp_pdist_euclidean_squareform = squareform(pdist(nonlinear_max_2_rp_samples, 'euclidean'));



  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %                                 GET PROJECTED ANGULAR AND EUCLIDEAN DISTANCES
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  [nonlinear_max_2_rp_angular_distances, nonlinear_max_2_rp_euclidean_distances] = tmpFunction( ...
    nonlinear_max_2_rp_pdist_angular_squareform, ...
    nonlinear_max_2_rp_pdist_euclidean_squareform, ...
    sample_pairs_with_original_angle_less_than_threshold, ...
    sample_pairs_with_original_angle_more_than_threshold);














  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %                                                                 MAX 8 PROJECT
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  max_count = 8;
  nonlinear_max_8_rp_samples = original_samples * randn(sample_dim, max_count * sample_dim) / sqrt(sample_dim);
  % nonlinear_max_8_rp_samples(nonlinear_max_8_rp_samples < 0) = 0;
  tmp_nonlinear_max_8_rp_samples = zeros(size(original_samples, 1), size(original_samples, 2));
  for j = 1 : size(tmp_nonlinear_max_8_rp_samples, 1)
    for i = 1 : size(tmp_nonlinear_max_8_rp_samples, 2)
      % nonlinear_max_8_rp_samples(j, (i-1)*max_count+1:(i-1)*max_count+1+max_count-1)
      % keyboard
      tmp_nonlinear_max_8_rp_samples(j,i) = max(nonlinear_max_8_rp_samples(j, (i-1)*max_count+1:(i-1)*max_count+1+max_count-1));
    end
  end

  nonlinear_max_8_rp_pdist_angular_squareform = squareform(acosd(1 - pdist(nonlinear_max_8_rp_samples, 'cosine')));
  nonlinear_max_8_rp_pdist_euclidean_squareform = squareform(pdist(nonlinear_max_8_rp_samples, 'euclidean'));



  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %                                 GET PROJECTED ANGULAR AND EUCLIDEAN DISTANCES
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  [nonlinear_max_8_rp_angular_distances, nonlinear_max_8_rp_euclidean_distances] = tmpFunction( ...
    nonlinear_max_8_rp_pdist_angular_squareform, ...
    nonlinear_max_8_rp_pdist_euclidean_squareform, ...
    sample_pairs_with_original_angle_less_than_threshold, ...
    sample_pairs_with_original_angle_more_than_threshold);





















  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %                                                                 MAX 32 PROJECT
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  max_count = 32;
  nonlinear_max_32_rp_samples = original_samples * randn(sample_dim, max_count * sample_dim) / sqrt(sample_dim);
  % nonlinear_max_32_rp_samples(nonlinear_max_32_rp_samples < 0) = 0;
  tmp_nonlinear_max_32_rp_samples = zeros(size(original_samples, 1), size(original_samples, 2));
  for j = 1 : size(tmp_nonlinear_max_32_rp_samples, 1)
    for i = 1 : size(tmp_nonlinear_max_32_rp_samples, 2)
      % nonlinear_max_32_rp_samples(j, (i-1)*max_count+1:(i-1)*max_count+1+max_count-1)
      % keyboard
      tmp_nonlinear_max_32_rp_samples(j,i) = max(nonlinear_max_32_rp_samples(j, (i-1)*max_count+1:(i-1)*max_count+1+max_count-1));
    end
  end

  nonlinear_max_32_rp_pdist_angular_squareform = squareform(acosd(1 - pdist(nonlinear_max_32_rp_samples, 'cosine')));
  nonlinear_max_32_rp_pdist_euclidean_squareform = squareform(pdist(nonlinear_max_32_rp_samples, 'euclidean'));



  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %                                 GET PROJECTED ANGULAR AND EUCLIDEAN DISTANCES
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  [nonlinear_max_32_rp_angular_distances, nonlinear_max_32_rp_euclidean_distances] = tmpFunction( ...
    nonlinear_max_32_rp_pdist_angular_squareform, ...
    nonlinear_max_32_rp_pdist_euclidean_squareform, ...
    sample_pairs_with_original_angle_less_than_threshold, ...
    sample_pairs_with_original_angle_more_than_threshold);
































  figure,


  subplot(6,2,1)
  subplotBeef(original_angular_distances, linspace(0, 180, 100), sample_dim, 'Original Angular Distances');

  subplot(6,2,2)
  subplotBeef(original_euclidean_distances, linspace(0, ceil(10*sqrt(sample_dim)), 100), sample_dim, 'Original Euclidean Distances');

  subplot(6,2,3)
  subplotBeef(linear_rp_angular_distances, linspace(0, 180, 100), sample_dim, 'Linear RP Angular Distances');

  subplot(6,2,4)
  subplotBeef(linear_rp_euclidean_distances, linspace(0, ceil(10*sqrt(sample_dim)), 100), sample_dim, 'Linear RP Euclidean Distances');

  subplot(6,2,5)
  subplotBeef(nonlinear_rectified_rp_angular_distances, linspace(0, 180, 100), sample_dim, 'Nonlinear Rectified RP Angular Distances');

  subplot(6,2,6)
  subplotBeef(nonlinear_rectified_rp_euclidean_distances, linspace(0, ceil(10*sqrt(sample_dim)), 100), sample_dim, 'Nonlinear Rectified RP Euclidean Distances');

  subplot(6,2,7)
  subplotBeef(nonlinear_max_2_rp_angular_distances, linspace(0, 180, 100), sample_dim, 'Nonlinear Max 2 RP Angular Distances');

  subplot(6,2,8)
  subplotBeef(nonlinear_max_2_rp_euclidean_distances, linspace(0, ceil(10*sqrt(sample_dim)), 100), sample_dim, 'Nonlinear Max 2 RP Euclidean Distances');

  subplot(6,2,9)
  subplotBeef(nonlinear_max_8_rp_angular_distances, linspace(0, 180, 100), sample_dim, 'Nonlinear Max 8 RP Angular Distances');

  subplot(6,2,10)
  subplotBeef(nonlinear_max_8_rp_euclidean_distances, linspace(0, ceil(10*sqrt(sample_dim)), 100), sample_dim, 'Nonlinear Max 8 RP Euclidean Distances');

  subplot(6,2,11)
  subplotBeef(nonlinear_max_32_rp_angular_distances, linspace(0, 180, 100), sample_dim, 'Nonlinear Max 32 RP Angular Distances');

  subplot(6,2,12)
  subplotBeef(nonlinear_max_32_rp_euclidean_distances, linspace(0, ceil(10*sqrt(sample_dim)), 100), sample_dim, 'Nonlinear Max 32 RP Euclidean Distances');




% -------------------------------------------------------------------------
function [angular_distances, euclidean_distances] = tmpFunction(pdist_angular_squareform, pdist_euclidean_squareform, sample_pairs_with_original_angle_less_than_threshold, sample_pairs_with_original_angle_more_than_threshold)
% -------------------------------------------------------------------------
  angular_distances = {};
  euclidean_distances = {};

  angular_distances.sample_pairs_with_original_angle_less_than_threshold = [];
  euclidean_distances.sample_pairs_with_original_angle_less_than_threshold = [];

  angular_distances.sample_pairs_with_original_angle_more_than_threshold = [];
  euclidean_distances.sample_pairs_with_original_angle_more_than_threshold = [];

  for k = 1 : numel(sample_pairs_with_original_angle_less_than_threshold)
    pair = sample_pairs_with_original_angle_less_than_threshold{k};
    index_j = pair(1);
    index_i = pair(2);
    angular_distances.sample_pairs_with_original_angle_less_than_threshold(end+1) = pdist_angular_squareform(index_j,index_i);
    euclidean_distances.sample_pairs_with_original_angle_less_than_threshold(end+1) = pdist_euclidean_squareform(index_j,index_i);
  end

  for k = 1 : numel(sample_pairs_with_original_angle_more_than_threshold)
    pair = sample_pairs_with_original_angle_more_than_threshold{k};
    index_j = pair(1);
    index_i = pair(2);
    angular_distances.sample_pairs_with_original_angle_more_than_threshold(end+1) = pdist_angular_squareform(index_j,index_i);
    euclidean_distances.sample_pairs_with_original_angle_more_than_threshold(end+1) = pdist_euclidean_squareform(index_j,index_i);
  end


% -------------------------------------------------------------------------
function subplotBeef(data_struct, x_ticks, sample_dim, title_string)
% -------------------------------------------------------------------------

  color_palette = {'g', 'c', 'b', 'k', 'r'};

  % figure,

  % subplot(2,2,1)
  title(title_string)
  hold on
  histogram(data_struct.sample_pairs_with_original_angle_less_than_threshold, x_ticks, 'facecolor', color_palette{1});
  histogram(data_struct.sample_pairs_with_original_angle_more_than_threshold, x_ticks, 'facecolor', color_palette{2});
  hold off


