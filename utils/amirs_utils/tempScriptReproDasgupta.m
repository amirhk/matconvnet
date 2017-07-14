% -------------------------------------------------------------------------
function tempScriptReproDasgupta(metric)
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
  original_dim_list = ['whatever'];

  % projected_dim_list = 25:1:50;
  % projected_dim_list = [10, 20, 40, 80, 160, 320];
  % projected_dim_list = [10, 20, 40, 80, 160];
  % projected_dim_list = [10, 20];
  % projected_dim_list = [10];
  % projected_dim_list = 100:100:1000;
  projected_dim_list = 100:100:300;

  % number_of_samples_list = 25:25:500;
  % number_of_samples_list = [10000];
  % number_of_samples_list = [100, 250, 500, 1000, 2500, 5000, 10000, 25000];
  % number_of_samples_list = [100, 250, 500, 1000, 2500];
  % number_of_samples_list = [100, 250];
  % number_of_samples_list = [10000, 25000, 50000, 100000];
  number_of_samples_list = [10, 50, 100, 250, 500, 1000, 2500];
  number_of_samples_list = [10, 50];

  % metric = 'measure-c-separation';
  % metric = 'measure-eccentricity';
  % metric = 'measure-1-knn-perf';
  % metric = 'measure-linear-svm-perf';
  % metric = 'measure-mlp-500-100-perf';

  fh_projection_utils = projectionUtils;

  repeat_count = 3;

  dataset = '2_gaussians';
  dataset = 'cifar';
  c_separation = 1;
  eccentricity = 1000;

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
  proj_w_tanh_imdb_results_mean = zeros(results_size);
  proj_w_tanh_imdb_results_std = zeros(results_size);
  proj_w_sigmoid_imdb_results_mean = zeros(results_size);
  proj_w_sigmoid_imdb_results_std = zeros(results_size);

  counter = 1;
  for original_dim = original_dim_list

    for projected_dim = projected_dim_list

      for number_of_samples = number_of_samples_list

        afprintf(sprintf('[INFO] Constructing imdbs...\n'));
        tmp_orig_imdb_results = [];
        tmp_proj_wo_non_lin_imdb_results = [];
        tmp_proj_w_relu_imdb_results = [];
        tmp_proj_w_tanh_imdb_results = [];
        tmp_proj_w_sigmoid_imdb_results = [];

        for j = 1 : repeat_count

          switch dataset
            case '2_gaussians'
              afprintf(sprintf('[INFO] Created new imdb...\n'));
              original_imdb = constructSyntheticGaussianImdbNEW(number_of_samples, original_dim, c_separation, eccentricity);
              afprintf(sprintf('[INFO] done!\n'));
            case 'cifar'
              afprintf(sprintf('[INFO] Loading original imdb...\n'));
              tmp_opts.dataset = dataset;
              tmp_opts.posneg_balance = number_of_samples;
              original_imdb = loadSavedImdb(tmp_opts, 1);
              afprintf(sprintf('[INFO] done!\n'));
            otherwise
              throwException('[ERROR] dataset not defined!');
          end

          projected_imdb_wo_non_lin = fh_projection_utils.getDenslyDownProjectedImdb(original_imdb, 1, 0, projected_dim, 'relu');
          projected_imdb_w_relu = fh_projection_utils.getDenslyDownProjectedImdb(original_imdb, 1, 1, projected_dim, 'relu');
          projected_imdb_w_tanh = fh_projection_utils.getDenslyDownProjectedImdb(original_imdb, 1, 1, projected_dim, 'tanh');
          projected_imdb_w_sigmoid = fh_projection_utils.getDenslyDownProjectedImdb(original_imdb, 1, 1, projected_dim, 'sigmoid');

          switch metric
            case 'measure-c-separation'
              tmp_orig_imdb_results(end+1) = getTwoClassCSeparation(original_imdb);
              tmp_proj_wo_non_lin_imdb_results(end+1) = getTwoClassCSeparation(projected_imdb_wo_non_lin);
              tmp_proj_w_relu_imdb_results(end+1) = getTwoClassCSeparation(projected_imdb_w_relu);
              tmp_proj_w_tanh_imdb_results(end+1) = getTwoClassCSeparation(projected_imdb_w_tanh);
              tmp_proj_w_sigmoid_imdb_results(end+1) = getTwoClassCSeparation(projected_imdb_w_sigmoid);
            case 'measure-eccentricity'
              tmp_orig_imdb_results(end+1) = getAverageClassEccentricity(original_imdb);
              tmp_proj_wo_non_lin_imdb_results(end+1) = getAverageClassEccentricity(projected_imdb_wo_non_lin);
              tmp_proj_w_relu_imdb_results(end+1) = getAverageClassEccentricity(projected_imdb_w_relu);
              tmp_proj_w_tanh_imdb_results(end+1) = getAverageClassEccentricity(projected_imdb_w_tanh);
              tmp_proj_w_sigmoid_imdb_results(end+1) = getAverageClassEccentricity(projected_imdb_w_sigmoid);
            case 'measure-1-knn-perf'
              tmp_orig_imdb_results(end+1) = getSimpleTestAccuracyFrom1Knn(original_imdb);
              tmp_proj_wo_non_lin_imdb_results(end+1) = getSimpleTestAccuracyFrom1Knn(projected_imdb_wo_non_lin);
              tmp_proj_w_relu_imdb_results(end+1) = getSimpleTestAccuracyFrom1Knn(projected_imdb_w_relu);
              tmp_proj_w_tanh_imdb_results(end+1) = getSimpleTestAccuracyFrom1Knn(projected_imdb_w_tanh);
              tmp_proj_w_sigmoid_imdb_results(end+1) = getSimpleTestAccuracyFrom1Knn(projected_imdb_w_sigmoid);
            case 'measure-linear-svm-perf'
              tmp_orig_imdb_results(end+1) = getSimpleTestAccuracyFromLibSvm(original_imdb);
              tmp_proj_wo_non_lin_imdb_results(end+1) = getSimpleTestAccuracyFromLibSvm(projected_imdb_wo_non_lin);
              tmp_proj_w_relu_imdb_results(end+1) = getSimpleTestAccuracyFromLibSvm(projected_imdb_w_relu);
              tmp_proj_w_tanh_imdb_results(end+1) = getSimpleTestAccuracyFromLibSvm(projected_imdb_w_tanh);
              tmp_proj_w_sigmoid_imdb_results(end+1) = getSimpleTestAccuracyFromLibSvm(projected_imdb_w_sigmoid);
            case 'measure-mlp-500-100-perf'
              tmp_orig_imdb_results(end+1) = getSimpleTestAccuracyFromMLP(original_imdb);
              tmp_proj_wo_non_lin_imdb_results(end+1) = getSimpleTestAccuracyFromMLP(projected_imdb_wo_non_lin);
              tmp_proj_w_relu_imdb_results(end+1) = getSimpleTestAccuracyFromMLP(projected_imdb_w_relu);
              tmp_proj_w_tanh_imdb_results(end+1) = getSimpleTestAccuracyFromMLP(projected_imdb_w_tanh);
              tmp_proj_w_sigmoid_imdb_results(end+1) = getSimpleTestAccuracyFromMLP(projected_imdb_w_sigmoid);
          end

        end

        orig_imdb_results_mean(counter) = mean(tmp_orig_imdb_results);
        orig_imdb_results_std(counter) = std(tmp_orig_imdb_results);
        proj_wo_non_lin_imdb_results_mean(counter) = mean(tmp_proj_wo_non_lin_imdb_results);
        proj_wo_non_lin_imdb_results_std(counter) = std(tmp_proj_wo_non_lin_imdb_results);
        proj_w_relu_imdb_results_mean(counter) = mean(tmp_proj_w_relu_imdb_results);
        proj_w_relu_imdb_results_std(counter) = std(tmp_proj_w_relu_imdb_results);
        proj_w_tanh_imdb_results_mean(counter) = mean(tmp_proj_w_tanh_imdb_results);
        proj_w_tanh_imdb_results_std(counter) = std(tmp_proj_w_tanh_imdb_results);
        proj_w_sigmoid_imdb_results_mean(counter) = mean(tmp_proj_w_sigmoid_imdb_results);
        proj_w_sigmoid_imdb_results_std(counter) = std(tmp_proj_w_sigmoid_imdb_results);


        counter = counter + 1;

      end

    end

  end

  h = figure;

  subplot(1,5,1),
  title_string = 'Orig. Imdb';
  subplotBeef(orig_imdb_results_mean, title_string, x_label, y_label, x_lim, y_lim, x_tick_lables, y_tick_lables, metric);

  subplot(1,5,2),
  title_string = 'Proj. Imdb - RP 1';
  subplotBeef(proj_wo_non_lin_imdb_results_mean, title_string, x_label, y_label, x_lim, y_lim, x_tick_lables, y_tick_lables, metric);

  subplot(1,5,3),
  title_string = 'Proj. Imdb - RP 1 RELU 1';
  subplotBeef(proj_w_relu_imdb_results_mean, title_string, x_label, y_label, x_lim, y_lim, x_tick_lables, y_tick_lables, metric);

  subplot(1,5,4),
  title_string = 'Proj. Imdb - RP 1 TANH 1';
  subplotBeef(proj_w_tanh_imdb_results_mean, title_string, x_label, y_label, x_lim, y_lim, x_tick_lables, y_tick_lables, metric);

  subplot(1,5,5),
  title_string = 'Proj. Imdb - RP 1 SIGMOID 1';
  subplotBeef(proj_w_sigmoid_imdb_results_mean, title_string, x_label, y_label, x_lim, y_lim, x_tick_lables, y_tick_lables, metric);


  plot_title = sprintf( ...
    '%s - c-sep = %d - ecc = %d - %s - %s', ...
    strrep(dataset,'_',' '), ...
    c_separation * 100, ...
    eccentricity * 100, ...
    sup_title, ...
    metric);
    % upper(strrep(random_projection_type,'_',' '))
  suptitle(plot_title);
  print(fullfile(getDevPath(), 'temp_images', plot_title), '-dpdf', '-fillpage')
  savefig(h, fullfile(getDevPath(), 'temp_images', plot_title));

% -------------------------------------------------------------------------
function subplotBeef(data, title_string, x_label, y_label, x_lim, y_lim, x_tick_lables, y_tick_lables, metric)
% -------------------------------------------------------------------------
  mesh(data),
  title(title_string),
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




















