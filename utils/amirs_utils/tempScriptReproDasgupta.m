% -------------------------------------------------------------------------
function tempScriptReproDasgupta()
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
  % projected_dim_list = 25:1:50;
  % number_of_samples_list = 25:25:500;
  % original_dim_list = 25:25:500;
  original_dim_list = 100;
  % projected_dim_list = [10, 20, 40, 80, 160, 320];
  % projected_dim_list = [10, 20, 40, 80, 160];
  projected_dim_list = [10, 20];
  % number_of_samples_list = [100, 250, 500, 1000, 2500, 5000];
  number_of_samples_list = [100, 250, 500, 1000, 2500];
  % number_of_samples_list = [100, 250];

  % test_type = 'vary_original_dim';
  % test_type = 'vary_projected_dim';
  % test_type = 'vary_number_of_samples';

  metric = 'measure-c-separation';
  % metric = 'measure-eccentricity';
  % metric = 'measure-1-knn-perf';
  % metric_list = {'c_separation', 'eccentricity', '1-knn'};
  % metric_list = {'c_separation', '1-knn'};
  % metric_list = {'c_separation'};
  % metric_list = {'eccentricity'};
  % for metric = metric_list

  fh_projection_utils = projectionUtils;

  repeat_count = 30;
  c_separation = 1;
  eccentricity = 1;

  assert( ...
    length(original_dim_list) == 1 || ...
    length(projected_dim_list) == 1 || ...
    length(number_of_samples_list) == 1);

  if length(original_dim_list) == 1
    results_size = [length(number_of_samples_list), length(projected_dim_list)];
    x_label = 'projected dim';
    y_label = 'num samples';
    x_tick_lables = projected_dim_list;
    y_tick_lables = number_of_samples_list;
    x_lim = [1 - 0.2, length(projected_dim_list) + 0.2];
    y_lim = [1 - 0.2, length(number_of_samples_list) + 0.2];
  elseif length(projected_dim_list) == 1
    results_size = [length(number_of_samples_list), length(original_dim_list)];
    x_label = 'original dim';
    y_label = 'num samples';
    x_tick_lables = original_dim_list;
    y_tick_lables = number_of_samples_list;
    x_lim = [1 - 0.2, length(original_dim_list) + 0.2];
    y_lim = [1 - 0.2, length(number_of_samples_list) + 0.2];
  elseif length(number_of_samples_list) == 1
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
  proj_imdb_results_mean = zeros(results_size);
  proj_imdb_results_std = zeros(results_size);

  counter = 1;
  for original_dim = original_dim_list

    for projected_dim = projected_dim_list

      for number_of_samples = number_of_samples_list

        afprintf(sprintf('[INFO] Constructing imdbs...\n'));
        tmp_orig_imdb_results = [];
        tmp_proj_imdb_results = [];

        for j = 1 : repeat_count

          original_imdb = constructSyntheticGaussianImdbNEW(number_of_samples, original_dim, 1, 1);
          projected_imdb = fh_projection_utils.getDenslyDownProjectedImdb(original_imdb, 1, 0, projected_dim);

          switch metric
            case 'measure-c-separation'
              tmp_orig_imdb_results(end+1) = getTwoClassCSeparation(original_imdb);
              tmp_proj_imdb_results(end+1) = getTwoClassCSeparation(projected_imdb);
            case 'measure-eccentricity'
              tmp_orig_imdb_results(end+1) = getAverageClassEccentricity(original_imdb);
              tmp_proj_imdb_results(end+1) = getAverageClassEccentricity(projected_imdb);
            case 'measure-1-knn-perf'
              tmp_orig_imdb_results(end+1) = get1KnnTestAccuracy(original_imdb);
              tmp_proj_imdb_results(end+1) = get1KnnTestAccuracy(projected_imdb);
          end

        end

        orig_imdb_results_mean(counter) = mean(tmp_orig_imdb_results);
        orig_imdb_results_std(counter) = std(tmp_orig_imdb_results);
        proj_imdb_results_mean(counter) = mean(tmp_proj_imdb_results);
        proj_imdb_results_std(counter) = std(tmp_proj_imdb_results);

      end

    end

  end

  figure,
  subplot(1,2,1),
  subplotBeef(orig_imdb_results_mean, 'Orig. Imdb', x_label, y_label, x_lim, y_lim, x_tick_lables, y_tick_lables, metric);
  subplot(1,2,2),
  subplotBeef(proj_imdb_results_mean, 'Proj. Imdb', x_label, y_label, x_lim, y_lim, x_tick_lables, y_tick_lables, metric);

% -------------------------------------------------------------------------
function subplotBeef(data, title_string, x_label, y_label, x_lim, y_lim, x_tick_lables, y_tick_lables, metric)
% -------------------------------------------------------------------------
  mesh(data),
  title(title_string),
  xlabel(x_label),
  ylabel(y_label),
  % xlim(x_lim),
  % ylim(y_lim),
  xticks(1:1:length(x_tick_lables)),
  yticks(1:1:length(y_tick_lables)),
  xticklabels(x_tick_lables),
  yticklabels(y_tick_lables),
  zlabel(metric);

%   zlabel(metric);

%         for i = 1 : numel(experiments)
%           switch metric
%             case 'measure-c-separation'
%               arr_1(j,i) = getTwoClassCSeparation(experiments{i}.original_imdb);
%               arr_2(j,i) = getTwoClassCSeparation(experiments{i}.projected_imdb);
%             case 'measure-eccentricity'
%               arr_1(j,i) = getAverageClassEccentricity(experiments{i}.original_imdb);
%               arr_2(j,i) = getAverageClassEccentricity(experiments{i}.projected_imdb);
%             case 'measure-1-knn-perf'
%               arr_1(j,i) = get1KnnTestAccuracy(experiments{i}.original_imdb);
%               arr_2(j,i) = get1KnnTestAccuracy(experiments{i}.projected_imdb);
%           end
%         end







%         % metric = char(metric);
%         arr_1 = [];
%         arr_2 = [];

%         for j = 1 : repeat_count

%           experiments = {};

%           afprintf(sprintf('[INFO] Constructing imdbs...\n'));

%           switch test_type

%             case 'vary_original_dim'

%               % 1-separated (also change constructSyntheticGaussianImdb.m)
%               experiment_title = sprintf('D = vary; d = %d; N = %d; C-Sep: 1; Ecc = 1', projected_dim, number_of_samples);
%               x_label_string = 'Orig. Dim.';
%               x_tick_lables = original_dim_list;

%               for original_dim = original_dim_list
%                 % projected_dim = 20;
%                 experiments{end+1}.original_imdb = constructSyntheticGaussianImdbNEW(number_of_samples, original_dim, 1, 1);
%                 experiments{end}.projected_imdb = fh_projection_utils.getDenslyDownProjectedImdb(experiments{end}.original_imdb, 1, 0, projected_dim);
%               end


%             case 'vary_projected_dim'

%               % Eccentricity 1000 (also change constructSyntheticGaussianImdb.m)
%               experiment_title = 'D = 50 - d = vary - N = 10000 - C-Sep: 1; Ecc = 1000';
%               x_label_string = 'Proj. Dim.';
%               x_tick_lables = projected_dim_list;

%               for projected_dim = projected_dim_list
%                 experiments{end+1}.original_imdb = constructSyntheticGaussianImdbNEW(10000, 50, 1, 1000);
%                 experiments{end}.projected_imdb = fh_projection_utils.getDenslyDownProjectedImdb(experiments{end}.original_imdb, 1, 0, projected_dim);
%               end


%             case 'vary_number_of_samples'

%               % 1-separated (also change constructSyntheticGaussianImdb.m)
%               experiment_title = 'D = 100 - d = 10log(N) - N = 10000 - C-Sep: 1; Ecc = 1';
%               x_label_string = 'Num. of Samples';
%               x_tick_lables = number_of_samples_list;

%               for number_of_samples = number_of_samples_list
%                 % projected_dim = ceil(10);
%                 % projected_dim = ceil(5 * log(number_of_samples));
%                 projected_dim = ceil(10 * log(number_of_samples));
%                 % projected_dim = ceil(25 * log(number_of_samples));
%                 experiments{end+1}.original_imdb = constructSyntheticGaussianImdbNEW(number_of_samples, 10000, 1, 1);
%                 experiments{end}.projected_imdb = fh_projection_utils.getDenslyDownProjectedImdb(experiments{end}.original_imdb, 1, 0, projected_dim);
%               end

%           end

%           afprintf(sprintf('done!\n\n'));
%           afprintf(sprintf('[INFO] Computing metrics...\n'));

%           for i = 1 : numel(experiments)
%             switch metric
%               case 'c_separation'
%                 experiments{i}.c_separation.original_imdb = getTwoClassCSeparation(experiments{i}.original_imdb);
%                 experiments{i}.c_separation.projected_imdb = getTwoClassCSeparation(experiments{i}.projected_imdb);
%                 arr_1(j,i) = experiments{i}.c_separation.original_imdb;
%                 arr_2(j,i) = experiments{i}.c_separation.projected_imdb;
%               case 'eccentricity'
%                 experiments{i}.average_class_eccentricity.original_imdb = getAverageClassEccentricity(experiments{i}.original_imdb);
%                 experiments{i}.average_class_eccentricity.projected_imdb = getAverageClassEccentricity(experiments{i}.projected_imdb);
%                 arr_1(j,i) = experiments{i}.average_class_eccentricity.original_imdb;
%                 arr_2(j,i) = experiments{i}.average_class_eccentricity.projected_imdb;
%               case '1-knn'
%                 experiments{i}.one_knn_test_accuracy.original_imdb = get1KnnTestAccuracy(experiments{i}.original_imdb);
%                 experiments{i}.one_knn_test_accuracy.projected_imdb = get1KnnTestAccuracy(experiments{i}.projected_imdb);
%                 arr_1(j,i) = experiments{i}.one_knn_test_accuracy.original_imdb;
%                 arr_2(j,i) = experiments{i}.one_knn_test_accuracy.projected_imdb;
%             end
%           end
%           afprintf(sprintf('done!\n\n'));
%         end


%         % shared_y_upper_limit = max(max(mean(arr_1) + std(arr_1)), max(mean(arr_2) + std(arr_2))) * 1.05;
%         % shared_y_upper_limit = 1.2;
%         % subplot(numel(metric_list), 2, 1 + (counter - 1) * 2),
%         % subplotBeef(metric, 'Orig. Imdb', arr_1, 'bs-', x_label_string, x_tick_lables, shared_y_upper_limit);
%         % subplot(numel(metric_list), 2, 2 + (counter - 1) * 2),
%         % subplotBeef(metric, 'Proj. Imdb', arr_2, 'rs-', x_label_string, x_tick_lables, shared_y_upper_limit);

%         shared_y_upper_limit = 1.25;
%         subplot(length(number_of_samples_list), length(projected_dim_list), counter);
%         plotBeef(metric, arr_1, arr_2, experiment_title, x_label_string, x_tick_lables, shared_y_upper_limit);

%         counter = counter + 1;
%       end
%     end
%   end
%   % end

%   % plot_title = sprintf('Repro Dasgupta - %s', experiment_title);
%   plot_title = sprintf('Repro Dasgupta - %s', test_type);
%   suptitle(plot_title);
%   print(fullfile(getDevPath(), 'temp_images', plot_title), '-dpdf', '-fillpage')
%   % keyboard


% % -------------------------------------------------------------------------
% function plotBeef(metric, arr_1, arr_2, experiment_title, x_label_string, x_tick_lables, shared_y_upper_limit)
% % -------------------------------------------------------------------------
%   assert(size(arr_1, 1) == size(arr_2, 1));
%   assert(size(arr_1, 2) == size(arr_2, 2));
%   hold on,
%   errorbar(1:1:size(arr_1, 2), mean(arr_1), std(arr_1), 'b*-'),
%   errorbar(1:1:size(arr_2, 2), mean(arr_2), std(arr_2), 'rs-'),
%   ylim([0, shared_y_upper_limit]),
%   xlim([1 - 0.2, size(arr_1, 2) + 0.2]),
%   xticks(1:1:size(arr_1, 2)),
%   xticklabels(x_tick_lables),
%   xlabel(x_label_string, 'FontSize', 12);
%   ylabel(sprintf('%s', strrep(metric,'_',' ')), 'FontSize', 12);
%   legend({'Orig. Imdb', 'Proj. Imdb'}, 'Location','southeast');
%   title(sprintf('%s', strrep(experiment_title,'_',' ')), 'FontSize', 12);


% % % -------------------------------------------------------------------------
% % function subplotBeef(metric, imdb_string, arr, color, x_label_string, x_tick_lables, shared_y_upper_limit)
% % % -------------------------------------------------------------------------
% %   errorbar(1:1:size(arr, 2), mean(arr), std(arr), color),
% %   ylim([0, shared_y_upper_limit]),
% %   xlim([1 - 0.2, size(arr, 2) + 0.2]),
% %   xticks(1:1:size(arr, 2)),
% %   xticklabels(x_tick_lables),
% %   xlabel(x_label_string, 'FontSize', 12);
% %   ylabel(sprintf('%s', strrep(metric,'_',' ')), 'FontSize', 12);
% %   title(sprintf('%s', strrep(imdb_string,'_',' ')), 'FontSize', 12);













