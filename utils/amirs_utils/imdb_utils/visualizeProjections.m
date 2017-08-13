% -------------------------------------------------------------------------
function visualizeProjections(number_of_classes, data_dim);
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

  % number_of_classes = 3;
  % data_dim = 2;

  imdb = constructSyntheticGaussianImdbNEW(number_of_classes, 1000, data_dim, 2, 1, true);
  vectorized_imdb = getVectorizedImdb(imdb);
  number_of_samples = size(imdb.images.data, 4);


  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % No RP (Original Imdb)
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  data_original = vectorized_imdb.images.data;
  data_original_per_class = struct();
  for i = 1 : number_of_classes
    data_original_per_class.in_original_space.(sprintf('class_%d', i)) = vectorized_imdb.images.data(imdb.images.labels == i,:);
  end

  test_accuracy_original.in_original_space = get1NNAccuracy(imdb, data_original);




  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % Create RPs
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  optimal_projection_lines = zeros(number_of_classes, data_dim);
  for i = 1 : number_of_classes
    optimal_projection_lines(i,:) = mean(data_original_per_class.in_original_space.(sprintf('class_%d', i)));
  end
  optimal_projection_lines = normr(optimal_projection_lines);

  number_of_projection_lines = 4;
  projection_lines = normr(randn(number_of_projection_lines, data_dim));




  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % Single RP
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  data_projected_single_rp = struct();
  tmp_projected_line = projection_lines(1,:);
  data_projected_single_rp.in_original_space = (tmp_projected_line * data_original')' * tmp_projected_line;
  data_projected_single_rp.in_projected_space = (tmp_projected_line * data_original')';

  data_projected_single_rp_per_class = struct();
  for i = 1 : number_of_classes
    data_projected_single_rp_per_class.in_original_space.(sprintf('class_%d', i)) = data_projected_single_rp.in_original_space(imdb.images.labels == i,:);
    data_projected_single_rp_per_class.in_projected_space.(sprintf('class_%d', i)) = data_projected_single_rp.in_projected_space(imdb.images.labels == i,:);
  end

  test_accuracy_single_rp.in_original_space = get1NNAccuracy(imdb, data_projected_single_rp.in_original_space);
  test_accuracy_single_rp.in_projected_space = get1NNAccuracy(imdb, data_projected_single_rp.in_projected_space);




  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % Multiple RP
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  data_projected_multi_rp = struct();
  data_projected_multi_rp.in_original_space = [];
  data_projected_multi_rp.in_projected_space = [];
  for i = 1 : size(data_original, 1)
    projected_point_magnitudes = projection_lines * data_original(i,:)';

    [~, index_of_max_projected_point_magnitude] = max(projected_point_magnitudes);

    % projected_line below is projected_line_giving_max_magnitude
    tmp_projected_line = projection_lines(index_of_max_projected_point_magnitude, :);
    data_projected_multi_rp.in_original_space(end+1,:) = (tmp_projected_line * data_original(i,:)') * tmp_projected_line;
    data_projected_multi_rp.in_projected_space(end+1,:) = tmp_projected_line * data_original(i,:)';
  end

  data_projected_multi_rp_per_class = struct();
  for i = 1 : number_of_classes
    data_projected_multi_rp_per_class.in_original_space.(sprintf('class_%d', i)) = data_projected_multi_rp.in_original_space(imdb.images.labels == i,:);
    data_projected_multi_rp_per_class.in_projected_space.(sprintf('class_%d', i)) = data_projected_multi_rp.in_projected_space(imdb.images.labels == i,:);
  end

  test_accuracy_multi_rp.in_original_space = get1NNAccuracy(imdb, data_projected_multi_rp.in_original_space);
  test_accuracy_multi_rp.in_projected_space = get1NNAccuracy(imdb, data_projected_multi_rp.in_projected_space);



  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % Multiple RP
  % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  data_projected_multi_optimal_rp = struct();
  data_projected_multi_optimal_rp.in_original_space = [];
  data_projected_multi_optimal_rp.in_projected_space = [];
  for i = 1 : size(data_original, 1)
    projected_point_magnitudes = optimal_projection_lines * data_original(i,:)';

    [~, index_of_max_projected_point_magnitude] = max(projected_point_magnitudes);

    % projected_line below is projected_line_giving_max_magnitude
    tmp_projected_line = optimal_projection_lines(index_of_max_projected_point_magnitude, :);
    data_projected_multi_optimal_rp.in_original_space(end+1,:) = (tmp_projected_line * data_original(i,:)') * tmp_projected_line;
    data_projected_multi_optimal_rp.in_projected_space(end+1,:) = tmp_projected_line * data_original(i,:)';
  end

  data_projected_multi_optimal_rp_per_class = struct();
  for i = 1 : number_of_classes
    data_projected_multi_optimal_rp_per_class.in_original_space.(sprintf('class_%d', i)) = data_projected_multi_optimal_rp.in_original_space(imdb.images.labels == i,:);
    data_projected_multi_optimal_rp_per_class.in_projected_space.(sprintf('class_%d', i)) = data_projected_multi_optimal_rp.in_projected_space(imdb.images.labels == i,:);
  end

  test_accuracy_multi_optimal_rp.in_original_space = get1NNAccuracy(imdb, data_projected_multi_optimal_rp.in_original_space);
  test_accuracy_multi_optimal_rp.in_projected_space = get1NNAccuracy(imdb, data_projected_multi_optimal_rp.in_projected_space);










  figure
  subplot(2,4,[1,5]),
  tmp_title = sprintf('Original Data - test acc: %.3f', test_accuracy_original.in_original_space);
  subplotScatterBeef(tmp_title, data_original_per_class, number_of_classes, projection_lines);

  subplot(2,4,2),
  tmp_title = sprintf('Projected Data (1 RP) - test acc: %.3f', test_accuracy_single_rp.in_original_space);
  subplotScatterBeef(tmp_title, data_projected_single_rp_per_class, number_of_classes, projection_lines(1,:));

  subplot(2,4,6),
  tmp_title = sprintf('Projected Data (1 RP) - test acc: %.3f', test_accuracy_single_rp.in_projected_space);
  subplotHistogramBeef(tmp_title, data_projected_single_rp_per_class, number_of_classes);

  subplot(2,4,3),
  tmp_title = sprintf('Projected Data (max %d RP) - test acc: %.3f', number_of_projection_lines, test_accuracy_multi_rp.in_original_space);
  subplotScatterBeef(tmp_title, data_projected_multi_rp_per_class, number_of_classes, projection_lines);

  subplot(2,4,7),
  tmp_title = sprintf('Projected Data (max %d RP) - test acc: %.3f', number_of_projection_lines, test_accuracy_multi_rp.in_projected_space);
  subplotHistogramBeef(tmp_title, data_projected_multi_rp_per_class, number_of_classes);

  subplot(2,4,4),
  tmp_title = sprintf('Projected Data (max %d optimal RP) - test acc: %.3f', number_of_projection_lines, test_accuracy_multi_optimal_rp.in_original_space);
  subplotScatterBeef(tmp_title, data_projected_multi_optimal_rp_per_class, number_of_classes, optimal_projection_lines);

  subplot(2,4,8),
  tmp_title = sprintf('Projected Data (max %d optimal RP) - test acc: %.3f', number_of_projection_lines, test_accuracy_multi_optimal_rp.in_projected_space);
  subplotHistogramBeef(tmp_title, data_projected_multi_optimal_rp_per_class, number_of_classes);











% -------------------------------------------------------------------------
function test_accuracy = get1NNAccuracy(imdb, data)
% -------------------------------------------------------------------------
  tmp_imdb = imdb;
  tmp_imdb.images.data = data;
  tmp_imdb = get4DImdb(tmp_imdb, size(data, 2), 1, 1, size(data, 1));
  tmp_opts = {};
  tmp_opts.imdb = tmp_imdb;
  test_accuracy = getSimpleTestAccuracyFromKnn(tmp_opts);






% -------------------------------------------------------------------------
function subplotScatterBeef(tmp_title, data_struct, number_of_classes, projection_lines)
% -------------------------------------------------------------------------
  color_palette = {'bs', 'ro', 'g*', 'bo', 'r*', 'gs'};

  title(tmp_title);
  if size(data_struct.in_original_space.class_1, 2) == 2
    hold on,
    grid on,
    for i = 1 : number_of_classes
      tmp = data_struct.in_original_space.(sprintf('class_%d', i));
      scatter(tmp(:,1), tmp(:,2), color_palette{i});
    end
    for i = 1 : size(projection_lines, 1)
      plot([0, projection_lines(i, 1)] * 5, [0, projection_lines(i, 2)] * 5, 'k--', 'LineWidth', 1.5);
    end
    hold off
    xlim([-5, 5]);
    ylim([-5, 5]);
  % elseif size(data_struct.class_1, 2) == 3
  %   hold on,
  %   grid on,
  %   for i = 1 : number_of_classes
  %     tmp = data_struct.(sprintf('class_%d', i));
  %     scatter3(tmp(:,1), tmp(:,2), tmp(:,3), color_palette{i});
  %   end
  %   for i = 1 : size(projection_lines, 1)
  %     plot([0, projection_lines(i, 1)] * 5, [0, projection_lines(i, 2)] * 5, [0, projection_lines(i, 3)] * 5, '--', 'LineWidth', 1.5);
  %   end
  %   hold off
  %   xlim([-5, 5]);
  %   ylim([-5, 5]);
  %   zlim([-5, 5]);
  else
    throwException('[ERROR] cannot support more than 3 dims.')
  end





% -------------------------------------------------------------------------
function subplotHistogramBeef(tmp_title, data_struct, number_of_classes);
% -------------------------------------------------------------------------
  color_palette = {'b', 'r', 'g', 'c'};

  title(tmp_title);
  hold on,
  grid on,
  x_ticks = -5:0.25:5;
  y_limits = [0 200];
  for i = 1 : number_of_classes
    data = data_struct.in_projected_space.(sprintf('class_%d', i));
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

























