% -------------------------------------------------------------------------
function tempScriptReproDasgupta(dataset, metric, c_separation, eccentricity)
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

  % original_dim_list = 25:25:500;
  % original_dim_list = 100:100:1000;
  % original_dim_list = 100:100:300;
  % original_dim_list = [1000];
  original_dim_list = [-1]; % cifar-multi-class-subsampled

  % projected_dim_list = 25:1:50;
  % projected_dim_list = [10, 20, 40, 80, 160, 320];
  % projected_dim_list = [10, 20, 40, 80, 160];
  % projected_dim_list = [10, 20];
  % projected_dim_list = [10];
  % projected_dim_list = 100:100:1000;
  % projected_dim_list = 100:100:300;
  % projected_dim_list = 10:10:100;
  % projected_dim_list = [10, 768:768:3072];
  projected_dim_list = [10, 768];

  % number_of_samples_list = [1000]; % 2_gaussians, 5_gaussians
  % number_of_samples_list = [10, 50, 100, 250, 500, 1000, 2500]; % circle_in_ring
  % number_of_samples_list = 10:10:100; % circle_in_ring
  % number_of_samples_list = [10, 50, 100, 250, 500, 1000, 2500]; % cifar-multi-class-subsampled
  % number_of_samples_list = [10, 50, 100, 250, 500]; % cifar-multi-class-subsampled
  number_of_samples_list = [10, 50, 100]; % cifar-multi-class-subsampled

  % metric = 'measure-c-separation';
  % metric = 'measure-eccentricity';
  % metric = 'measure-1-knn-perf';
  % metric = 'measure-linear-svm-perf';
  % metric = 'measure-mlp-500-100-perf';

  fh_projection_utils = projectionUtils;

  % repeat_count = 5;
  repeat_count = 2;

  % dataset = '2_gaussians';
  % dataset = '5_gaussians';
  % dataset = 'circle_in_ring';
  % dataset = 'cifar-multi-class-subsampled';

  % c_separation = 1;
  % eccentricity = 1;

  assert( ...
    length(original_dim_list) == 1 || ...
    length(projected_dim_list) == 1 || ...
    length(number_of_samples_list) == 1);

  if length(original_dim_list) == 1
    sup_title = sprintf('orig dim = %d', original_dim_list(1));
    results_size = [length(number_of_samples_list), length(projected_dim_list)];
    x_label = 'projected dim';
    y_label = 'num samples';
    x_tick_lables = projected_dim_list;
    y_tick_lables = number_of_samples_list;
    x_lim = [1 - 0.2, length(projected_dim_list) + 0.2];
    y_lim = [1 - 0.2, length(number_of_samples_list) + 0.2];
  elseif length(projected_dim_list) == 1
    sup_title = sprintf('proj dim = %d', projected_dim_list(1));
    results_size = [length(number_of_samples_list), length(original_dim_list)];
    x_label = 'original dim';
    y_label = 'num samples';
    x_tick_lables = original_dim_list;
    y_tick_lables = number_of_samples_list;
    x_lim = [1 - 0.2, length(original_dim_list) + 0.2];
    y_lim = [1 - 0.2, length(number_of_samples_list) + 0.2];
  elseif length(number_of_samples_list) == 1
    sup_title = sprintf('num samples = %d', number_of_samples_list(1));
    results_size = [length(projected_dim_list), length(original_dim_list)];
    x_label = 'original dim';
    y_label = 'projected dim';
    x_tick_lables = original_dim_list;
    y_tick_lables = projected_dim_list;
    x_lim = [1 - 0.2, length(original_dim_list) + 0.2];
    y_lim = [1 - 0.2, length(projected_dim_list) + 0.2];
  else
    throwException('[ERROR] can only vary 2 parameters!');
  end

  orig_imdb_results_mean = zeros(results_size);
  orig_imdb_results_std = zeros(results_size);
  proj_wo_non_lin_imdb_results_mean = zeros(results_size);
  proj_wo_non_lin_imdb_results_std = zeros(results_size);
  proj_w_relu_imdb_results_mean = zeros(results_size);
  proj_w_relu_imdb_results_std = zeros(results_size);

  counter = 1;

  h = figure;
  plot_title = sprintf( ...
    '%s - c-sep = %d - ecc = %d - %s - %s', ...
    strrep(dataset,'_',' '), ...
    c_separation * 100, ...
    eccentricity * 100, ...
    sup_title, ...
    metric);
    % upper(strrep(random_projection_type,'_',' '))
  suptitle(plot_title);

  for original_dim = original_dim_list

    for projected_dim = projected_dim_list

      for number_of_samples = number_of_samples_list

        afprintf(sprintf('[INFO] Test # %d / %d...\n', counter, length(original_dim_list) * length(projected_dim_list) * length(number_of_samples_list)), -1);

        tmp_orig_imdb_results = [];
        tmp_proj_wo_non_lin_imdb_results = [];
        tmp_proj_w_relu_imdb_results = [];

        for j = 1 : repeat_count

          afprintf(sprintf('[INFO] Created / loading new imdb...\n'));
          switch dataset
            case '2_gaussians'
              original_imdb = constructSyntheticGaussianImdbNEW(2, number_of_samples, original_dim, c_separation, eccentricity, true);
            case '5_gaussians'
              original_imdb = constructSyntheticGaussianImdbNEW(5, number_of_samples, original_dim, c_separation, eccentricity, true);
            case 'circle_in_ring'
              original_imdb = constructSyntheticCirclesImdb(number_of_samples, original_dim, 0, 1);
            case 'cifar-multi-class-subsampled'
              tmp_opts.dataset = dataset;
              tmp_opts.posneg_balance = sprintf('balanced-%d', number_of_samples);
              original_imdb = loadSavedImdb(tmp_opts, 0);
              % original_imdb = filterImdbForSet(original_imdb, 1, 1);
            otherwise
              throwException('[ERROR] dataset not recognized!');
          end
          afprintf(sprintf('[INFO] done!\n'));

          afprintf(sprintf('[INFO] Projecting imdb...\n'));
          projected_imdb_wo_non_lin = fh_projection_utils.getDenslyDownProjectedImdb(original_imdb, 1, 0, projected_dim, 'relu');
          projected_imdb_w_relu = fh_projection_utils.getDenslyDownProjectedImdb(original_imdb, 1, 1, projected_dim, 'relu');
          afprintf(sprintf('[INFO] done!\n'));

          afprintf(sprintf('[INFO] Evaluating metric...\n'));
          switch metric
            case 'measure-c-separation'
              tmp_orig_imdb_results(end+1) = getAverageClassCSeparation(original_imdb);
              tmp_proj_wo_non_lin_imdb_results(end+1) = getAverageClassCSeparation(projected_imdb_wo_non_lin);
              tmp_proj_w_relu_imdb_results(end+1) = getAverageClassCSeparation(projected_imdb_w_relu);
            case 'measure-eccentricity'
              tmp_orig_imdb_results(end+1) = getAverageClassEccentricity(original_imdb);
              tmp_proj_wo_non_lin_imdb_results(end+1) = getAverageClassEccentricity(projected_imdb_wo_non_lin);
              tmp_proj_w_relu_imdb_results(end+1) = getAverageClassEccentricity(projected_imdb_w_relu);
            case 'measure-1-knn-perf'
              tmp_orig_imdb_results(end+1) = getSimpleTestAccuracyFrom1Knn(original_imdb);
              tmp_proj_wo_non_lin_imdb_results(end+1) = getSimpleTestAccuracyFrom1Knn(projected_imdb_wo_non_lin);
              tmp_proj_w_relu_imdb_results(end+1) = getSimpleTestAccuracyFrom1Knn(projected_imdb_w_relu);
            case 'measure-linear-svm-perf'
              tmp_orig_imdb_results(end+1) = getSimpleTestAccuracyFromLibSvm(original_imdb);
              tmp_proj_wo_non_lin_imdb_results(end+1) = getSimpleTestAccuracyFromLibSvm(projected_imdb_wo_non_lin);
              tmp_proj_w_relu_imdb_results(end+1) = getSimpleTestAccuracyFromLibSvm(projected_imdb_w_relu);
            case 'measure-mlp-500-100-perf'
              tmp_orig_imdb_results(end+1) = getSimpleTestAccuracyFromMLP(original_imdb);
              tmp_proj_wo_non_lin_imdb_results(end+1) = getSimpleTestAccuracyFromMLP(projected_imdb_wo_non_lin);
              tmp_proj_w_relu_imdb_results(end+1) = getSimpleTestAccuracyFromMLP(projected_imdb_w_relu);
          end
          afprintf(sprintf('[INFO] done!\n'));

        end

        orig_imdb_results_mean(counter) = mean(tmp_orig_imdb_results);
        orig_imdb_results_std(counter) = std(tmp_orig_imdb_results);
        proj_wo_non_lin_imdb_results_mean(counter) = mean(tmp_proj_wo_non_lin_imdb_results);
        proj_wo_non_lin_imdb_results_std(counter) = std(tmp_proj_wo_non_lin_imdb_results);
        proj_w_relu_imdb_results_mean(counter) = mean(tmp_proj_w_relu_imdb_results);
        proj_w_relu_imdb_results_std(counter) = std(tmp_proj_w_relu_imdb_results);

        counter = counter + 1;

        afprintf(sprintf('[INFO] Updating subplots...\n'));

        subplot(1,3,1),
        title_string = 'Orig. Imdb';
        subplotBeef(orig_imdb_results_mean, title_string, x_label, y_label, x_lim, y_lim, x_tick_lables, y_tick_lables, metric);

        subplot(1,3,2),
        title_string = 'Proj. Imdb - RP 1';
        subplotBeef(proj_wo_non_lin_imdb_results_mean, title_string, x_label, y_label, x_lim, y_lim, x_tick_lables, y_tick_lables, metric);

        subplot(1,3,3),
        title_string = 'Proj. Imdb - RP 1 RELU 1';
        subplotBeef(proj_w_relu_imdb_results_mean, title_string, x_label, y_label, x_lim, y_lim, x_tick_lables, y_tick_lables, metric);

        afprintf(sprintf('[INFO] done!\n'));

      end

    end

  end


  keyboard
  afprintf(sprintf('[INFO] Saving plots...\n'));
  print(fullfile(getDevPath(), 'temp_images', plot_title), '-dpdf', '-fillpage')
  savefig(h, fullfile(getDevPath(), 'temp_images', plot_title));
  afprintf(sprintf('[INFO] done!\n'));

% -------------------------------------------------------------------------
function subplotBeef(data, title_string, x_label, y_label, x_lim, y_lim, x_tick_lables, y_tick_lables, metric)
% -------------------------------------------------------------------------
  mesh(data),
  title(sprintf('%s - V: %.3f', title_string, getVolumeUnderSurface(data))),
  xlabel(x_label),
  ylabel(y_label),
  zlabel(metric);
  xlim(x_lim),
  ylim(y_lim),
  zlim([0,1.2]),
  xticks(1:1:length(x_tick_lables)),
  yticks(1:1:length(y_tick_lables)),
  xticklabels(x_tick_lables),
  yticklabels(y_tick_lables),
  view(55,10);
  drawnow
  pause(0.05);


% -------------------------------------------------------------------------
function volume = getVolumeUnderSurface(z)
% -------------------------------------------------------------------------
  % keyboard
  % x = 1:0.1:size(z,2);
  % y = 1:0.1:size(z,1);
  x = 1:1:size(z,2);
  y = 1:1:size(z,1);
  volume = trapz(y, trapz(x, z, 2), 1);



















