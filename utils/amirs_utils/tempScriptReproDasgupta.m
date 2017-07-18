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

  % original_dim_list = 100:100:1000;
  original_dim_list = [3072]; % gaussian

  % projected_dim_list = [5, 10, 15, 20, 25, 50, 75, 100, 250, 500, 1000, 2000, 3072];
  projected_dim_list = [5, 10, 20, 100, 1000, 3072];

  % number_of_samples_list = [1000];                               % 2_gaussians, 5_gaussians
  % number_of_samples_list = [10, 50, 100, 250, 500, 1000, 2500];  % circle_in_ring
  number_of_samples_list = [10, 50, 100, 250, 500, 1000];          % all datasets, except for stl-10
  % number_of_samples_list = [10, 50, 100, 250, 500];              % only for stl-10

  counter = 1;
  repeat_count = 5;
  % repeat_count = 1;

  experiments_list = { ...
    'orig_imdb', ...
    'proj_imdb_rp_1_relu_0', ...
    'proj_imdb_rp_2_relu_0', ...
    'proj_imdb_rp_3_relu_0', ...
    'proj_imdb_rp_4_relu_0', ...
    'proj_imdb_rp_5_relu_0', ...
    'proj_imdb_rp_1_relu_1', ...
    'proj_imdb_rp_2_relu_2', ...
    'proj_imdb_rp_3_relu_3', ...
    'proj_imdb_rp_4_relu_4', ...
    'proj_imdb_rp_5_relu_5'};

  [ ...
    sup_title, ...
    results_size, ...
    x_label, ...
    y_label, ...
    x_tick_lables, ...
    y_tick_lables, ...
    x_lim, ...
    y_lim] = getResultsAndPlotSettings( ...
      original_dim_list, ...
      number_of_samples_list, ...
      projected_dim_list);

  global_results = {};
  for experiment = experiments_list
    experiment = char(experiment);
    global_results.(experiment).mean = zeros(results_size);
    global_results.(experiment).std = zeros(results_size);
  end

  for original_dim = original_dim_list

    for projected_dim = projected_dim_list

      for number_of_samples = number_of_samples_list

        afprintf(sprintf('[INFO] Test # %d / %d...\n', counter, length(original_dim_list) * length(projected_dim_list) * length(number_of_samples_list)), -1);

        tmp_results = {};
        for experiment = experiments_list
          experiment = char(experiment);
          tmp_results.(experiment) = [];
        end

        for j = 1 : repeat_count

          imdb_list = getImdbList(experiments_list, dataset, number_of_samples, original_dim, projected_dim, c_separation, eccentricity);

          afprintf(sprintf('[INFO] Evaluating metric on experiments...\n'));
          for experiment = experiments_list
            experiment = char(experiment);
            tmp_imdb = imdb_list.(experiment);
            switch metric
              case 'measure-c-separation'
                tmp_results.(experiment)(end+1) = getAverageClassCSeparation(tmp_imdb);
              case 'measure-eccentricity'
                tmp_results.(experiment)(end+1) = getAverageClassEccentricity(tmp_imdb);
              case 'measure-1-knn-perf'
                tmp_results.(experiment)(end+1) = getSimpleTestAccuracyFrom1Knn(tmp_imdb);
              case 'measure-linear-svm-perf'
                tmp_results.(experiment)(end+1) = getSimpleTestAccuracyFromLibSvm(tmp_imdb);
              case 'measure-mlp-500-100-perf'
                tmp_results.(experiment)(end+1) = getSimpleTestAccuracyFromMLP(tmp_imdb);
            end
          end
          afprintf(sprintf('[INFO] done!\n'));

        end

        for experiment = experiments_list
          experiment = char(experiment);
          global_results.(experiment).mean(counter) = mean(tmp_results.(experiment));
          global_results.(experiment).std(counter) = std(tmp_results.(experiment));
        end

        counter = counter + 1;

      end

    end

  end

  h = figure;

  afprintf(sprintf('[INFO] Plotting results...\n'));

  experiment_count = numel(experiments_list);
  assert( ...
    mod(experiment_count, 2) == 1, ...
    'experiments should contain `orig_imdb` and pairs of `proj_imdb` w/ and w/o non-linearities.');

  % plot for `orig_imdb`
  experiment = 'orig_imdb';
  subplot_y_length = 2;
  subplot_x_length = (experiment_count + 1) / 2;
  subplot(subplot_y_length, subplot_x_length, [1, 1 + subplot_x_length]),
  title_string = upper(strrep(experiment, '_', ' '));
  subplotBeef(global_results.(experiment).mean, title_string, x_label, y_label, x_lim, y_lim, x_tick_lables, y_tick_lables, metric);

  % plot for all experiments except `orig_imdb`
  counter = 2;
  for experiment = experiments_list
    experiment = char(experiment);
    if strcmp(experiment, 'orig_imdb')
      continue;
    end

    subplot(subplot_y_length, subplot_x_length, counter),
    title_string = upper(strrep(experiment, '_', ' '));
    subplotBeef(global_results.(experiment).mean, title_string, x_label, y_label, x_lim, y_lim, x_tick_lables, y_tick_lables, metric);

    if counter == subplot_x_length
      % going to next line in subplots... so incremement index by 2 instead of 1
      counter = counter + 2;
    else
      counter = counter + 1;
    end
  end

  afprintf(sprintf('[INFO] done!\n'));

  plot_title = sprintf( ...
    '%s - c-sep = %d - ecc = %d - %s - %s', ...
    strrep(dataset,'_',' '), ...
    c_separation * 100, ...
    eccentricity * 100, ...
    sup_title, ...
    metric);
    % upper(strrep(random_projection_type,'_',' '))
  suptitle(plot_title);

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
  view(25,10);
  % drawnow
  % pause(0.05);


% -------------------------------------------------------------------------
function volume = getVolumeUnderSurface(z)
% -------------------------------------------------------------------------
  % keyboard
  % x = 1:0.1:size(z,2);
  % y = 1:0.1:size(z,1);
  x = 1:1:size(z,2);
  y = 1:1:size(z,1);
  volume = trapz(y, trapz(x, z, 2), 1);


% -------------------------------------------------------------------------
function [sup_title, results_size, x_label, y_label, x_tick_lables, y_tick_lables, x_lim, y_lim] = getResultsAndPlotSettings(original_dim_list, number_of_samples_list, projected_dim_list)
% -------------------------------------------------------------------------
  assert( ...
    length(original_dim_list) == 1 || ...
    length(projected_dim_list) == 1 || ...
    length(number_of_samples_list) == 1, ...
    'at least one list must contain a single element, because we are plotting mesh in 3D (not 4D).');

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


% -------------------------------------------------------------------------
function imdb_list = getImdbList(experiments_list, dataset, number_of_samples, original_dim, projected_dim, c_separation, eccentricity)
% -------------------------------------------------------------------------
  fh_projection_utils = projectionUtils;
  imdb_list = {};
  afprintf(sprintf('[INFO] Created / loading new imdb...\n'));
  switch dataset
    case '2_gaussians'
      original_imdb = constructSyntheticGaussianImdbNEW(2, number_of_samples, original_dim, c_separation, eccentricity, true);
    case '5_gaussians'
      original_imdb = constructSyntheticGaussianImdbNEW(5, number_of_samples, original_dim, c_separation, eccentricity, true);
    case 'circle_in_ring'
      original_imdb = constructSyntheticCirclesImdb(number_of_samples, original_dim, 0, 1);
    otherwise
      if strcmp(dataset, 'cifar-multi-class-subsampled') || ...
        strcmp(dataset, 'cifar-no-white-multi-class-subsampled') || ...
        strcmp(dataset, 'stl-10-multi-class-subsampled') || ...
        strcmp(dataset, 'mnist-784-two-class-0-1') || ...
        strcmp(dataset, 'mnist-784-two-class-8-3') || ...
        strcmp(dataset, 'mnist-784-multi-class-subsampled') || ...
        strcmp(dataset, 'svhn-multi-class-subsampled')

        tmp_opts.dataset = dataset;
        tmp_opts.posneg_balance = sprintf('balanced-%d', number_of_samples);
        original_imdb = loadSavedImdb(tmp_opts, 0);

      else
        throwException('[ERROR] dataset not recognized!');
      end
  end
  imdb_list.('orig_imdb') = original_imdb;
  afprintf(sprintf('[INFO] done!\n'));

  afprintf(sprintf('[INFO] Projecting imdb...\n'));
  for experiment = experiments_list
    experiment = char(experiment);
    if strcmp(experiment, 'orig_imdb')
      continue;
    end
    number_of_projection_layers = str2num(experiment(strfind(experiment, 'rp_') + 3));
    number_of_non_linear_layers = str2num(experiment(strfind(experiment, 'relu_') + 5));
    imdb_list.(experiment) = fh_projection_utils.getDenslyDownProjectedImdb( ...
      original_imdb, ...
      number_of_projection_layers, ...
      number_of_non_linear_layers, ...
      projected_dim, ...
      'relu');
  end
  afprintf(sprintf('[INFO] done!\n'));










