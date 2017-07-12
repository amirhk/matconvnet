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

  original_dim_list = 25:25:500;
  % original_dim_list = [25, 50, 75, 100];
  projected_dim_list = 25:1:50;
  number_of_samples_list = 5:5:100;

  test_type = 'vary_original_dim';
  % test_type = 'vary_projected_dim';
  % test_type = 'vary_number_of_samples';

  % metric = 'c separation';
  % metric = 'eccentricity';
  % metric = '1-knn';
  metric_list = {'c separation', 'eccentricity', '1-knn'};
  % metric_list = {'c separation', '1-knn'};

  fh_projection_utils = projectionUtils;

  repeat_count = 30;
  counter = 1;
  figure;

  for metric = metric_list

    metric = char(metric);
    arr_1 = [];
    arr_2 = [];

    for j = 1 : repeat_count

      experiments = {};

      switch test_type

        case 'vary_original_dim'

          % 1-separated (also change constructSyntheticGaussianImdb.m)
          x_label_string = 'Varying Orig. Dim.';
          x_tick_lables = original_dim_list;

          for original_dim = original_dim_list
            projected_dim = 20;
            experiments{end+1}.original_imdb = constructSyntheticGaussianImdb(10000, original_dim, 1, 4, 1);
            experiments{end}.projected_imdb = fh_projection_utils.getDenslyDownProjectedImdb(experiments{end}.original_imdb, 1, 0, projected_dim);
          end


        case 'vary_projected_dim'

          % Eccentricity 1000 (also change constructSyntheticGaussianImdb.m)
          x_label_string = 'Varying Proj. Dim.';
          x_tick_lables = projected_dim_list;

          for projected_dim = projected_dim_list
            experiments{end+1}.original_imdb = constructSyntheticGaussianImdb(10000, 50, 1, 1, 1000);
            experiments{end}.projected_imdb = fh_projection_utils.getDenslyDownProjectedImdb(experiments{end}.original_imdb, 1, 0, projected_dim);
          end


        case 'vary_number_of_samples'

          % 1-separated (also change constructSyntheticGaussianImdb.m)
          x_label_string = 'Varying Num. of Samples';
          x_tick_lables = number_of_samples_list;

          for number_of_samples = number_of_samples_list
            % projected_dim = ceil(10);
            % projected_dim = ceil(5 * log(number_of_samples));
            projected_dim = ceil(10 * log(number_of_samples));
            % projected_dim = ceil(25 * log(number_of_samples));
            experiments{end+1}.original_imdb = constructSyntheticGaussianImdb(number_of_samples, 100, 1, 4, 1);
            experiments{end}.projected_imdb = fh_projection_utils.getDenslyDownProjectedImdb(experiments{end}.original_imdb, 1, 0, projected_dim);
          end

      end

      for i = 1 : numel(experiments)
        switch metric
          case 'c separation'
            experiments{i}.c_separation.original_imdb = getTwoClassCSeparation(experiments{i}.original_imdb);
            experiments{i}.c_separation.projected_imdb = getTwoClassCSeparation(experiments{i}.projected_imdb);
            arr_1(j,i) = experiments{i}.c_separation.original_imdb;
            arr_2(j,i) = experiments{i}.c_separation.projected_imdb;
          case 'eccentricity'
            experiments{i}.average_class_eccentricity.original_imdb = getAverageClassEccentricity(experiments{i}.original_imdb);
            experiments{i}.average_class_eccentricity.projected_imdb = getAverageClassEccentricity(experiments{i}.projected_imdb);
            arr_1(j,i) = experiments{i}.average_class_eccentricity.original_imdb;
            arr_2(j,i) = experiments{i}.average_class_eccentricity.projected_imdb;
          case '1-knn'
            experiments{i}.one_knn_test_accuracy.original_imdb = get1KnnTestAccuracy(experiments{i}.original_imdb);
            experiments{i}.one_knn_test_accuracy.projected_imdb = get1KnnTestAccuracy(experiments{i}.projected_imdb);
            arr_1(j,i) = experiments{i}.one_knn_test_accuracy.original_imdb;
            arr_2(j,i) = experiments{i}.one_knn_test_accuracy.projected_imdb;
        end
      end
    end

    subplot(numel(metric_list), 2, 1 + (counter - 1) * 2),
    subplotBeef(test_type, metric, 'Original Imdb', arr_1, 'bs-', x_label_string, x_tick_lables);
    subplot(numel(metric_list), 2, 2 + (counter - 1) * 2),
    subplotBeef(test_type, metric, 'Projected Imdb', arr_2, 'rs-', x_label_string, x_tick_lables);

    counter = counter + 1;
  end
  keyboard

% -------------------------------------------------------------------------
function subplotBeef(test_type, metric, additional_title_text, arr, color, x_label_string, x_tick_lables)
% -------------------------------------------------------------------------
  errorbar(1:1:size(arr, 2), mean(arr), std(arr), color),
  ylim([0, inf]),
  xlim([1 - 0.2, size(arr, 2) + 0.2]),
  xticks(1:1:size(arr, 2)),
  xticklabels(x_tick_lables),
  xlabel(x_label_string, 'FontSize', 20);
  ylabel(sprintf('%s - %s', metric, additional_title_text), 'FontSize', 20);













