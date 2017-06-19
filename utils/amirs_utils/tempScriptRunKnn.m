% -------------------------------------------------------------------------
function tempScriptRunKNN(dataset, posneg_balance)
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
  % dataset = 'cifar-multi-class-subsampled';
  % posneg_balance = 'balanced-266';
  % dataset = 'cifar-two-class-deer-truck';

  % dataset = 'gaussian-5D-160-train-40-test';
  % dataset = 'gaussian-10D-160-train-40-test';
  % dataset = 'gaussian-25D-160-train-40-test';
  % dataset = 'gaussian-50D-160-train-40-test';

  % dataset = 'gaussian-5D-400-train-100-test';
  % dataset = 'gaussian-10D-400-train-100-test';
  % dataset = 'gaussian-25D-400-train-100-test';
  % dataset = 'gaussian-50D-400-train-100-test';

  % dataset = 'gaussian-50D-800-train-200-test';
  % TODO: change name to include variance as well!!!!!!!!!!
  % generate more of the 0.1 variances and compare

  % dataset = 'gaussian-5D-160-train-40-test-0.1-var';
  % dataset = 'gaussian-5D-160-train-40-test-1.0-var';
  % dataset = 'gaussian-5D-160-train-40-test-10.0-var';

  % dataset = 'gaussian-5D-400-train-100-test-0.1-var';
  % dataset = 'gaussian-5D-400-train-100-test-1.0-var';
  % dataset = 'gaussian-5D-400-train-100-test-10.0-var';

  % dataset = 'gaussian-5D-800-train-200-test-0.1-var';
  % dataset = 'gaussian-5D-800-train-200-test-1.0-var';
  % dataset = 'gaussian-5D-800-train-200-test-10.0-var';

  % posneg_balance = 'balanced-38';

  repeat_count = 1;
  all_experiments_multi_run = {};

  for i = 1 : 22
    all_experiments_multi_run{i}.test_performance = [];
  end

  for kk = 1:repeat_count
    all_experiments_single_run = runAllExperimentsOnce(dataset, posneg_balance);
    for i = 1 : numel(all_experiments_single_run)
      all_experiments_multi_run{i}.test_performance(end + 1) = ...
        all_experiments_single_run{i}.performance_summary.testing.test.accuracy;
    end
  end


  % y = [];
  % std_errors_value = [];
  % exp_number = 1;
  % for j = 1:2
  %   for i = 1:11
  %     y(i,j) = mean(all_experiments_multi_run{exp_number}.test_performance);
  %     std_errors_value(end + 1) = std(all_experiments_multi_run{exp_number}.test_performance);
  %     exp_number = exp_number + 1;
  %   end
  % end

  % std_errors_x_location = [ ...
  %   0.86, 1.14, ...
  %   1.86, 2.14, ...
  %   2.86, 3.14, ...
  %   3.86, 4.14, ...
  %   4.86, 5.14, ...
  %   5.86, 6.14, ...
  %   6.86, 7.14, ...
  %   7.86, 8.14, ...
  %   8.86, 9.14, ...
  %   9.86, 10.14, ...
  %   10.86, 11.14];
  % std_errors_y_location = reshape(y', 1, []);

  % h = figure;
  % hold on
  % bar(y);
  % ylim([-0.5, 1.5]);
  % errorbar(std_errors_x_location, std_errors_y_location, std_errors_value);
  % tmp_string = sprintf('1-KNN - %s', dataset);
  % suptitle(tmp_string);
  % saveas(h, fullfile(getDevPath(), 'temp_images', sprintf('%s.png', tmp_string)));


  y_all = [];
  y_w_relu = [];
  y_wo_relu = [];
  std_errors_value_all = [];
  std_errors_value_w_relu = [];
  std_errors_value_wo_relu = [];
  exp_number = 1;
  for j = 1:2
    for i = 1:11
      y_all(i,j) = mean(all_experiments_multi_run{exp_number}.test_performance);
      std_errors_value_all(i,j) = std(all_experiments_multi_run{exp_number}.test_performance);
      exp_number = exp_number + 1;
    end
  end

  y_w_relu = [y_all(1,:); y_all(2:6,:)];
  y_wo_relu = [y_all(1,:); y_all(7:11,:)];

  std_errors_value_w_relu = reshape([std_errors_value_all(1,:); std_errors_value_all(2:6,:)]', 1, []);
  std_errors_value_wo_relu = reshape([std_errors_value_all(1,:); std_errors_value_all(7:11,:)]', 1, []);


  std_errors_x_location = [ ...
    0.86, 1.14, ...
    1.86, 2.14, ...
    2.86, 3.14, ...
    3.86, 4.14, ...
    4.86, 5.14, ...
    5.86, 6.14];
  std_errors_y_location = reshape(y_all', 1, []);
  std_errors_y_location_w_relu = cat(2, std_errors_y_location(1:2), std_errors_y_location(3:12));
  std_errors_y_location_wo_relu = cat(2, std_errors_y_location(1:2), std_errors_y_location(13:22));

  h = figure;

  subplot(1,2,1);
  hold on;
  bar(y_w_relu);
  ylim([-0.1, 1.1]);
  errorbar(std_errors_x_location, std_errors_y_location_w_relu, std_errors_value_w_relu);
  legend({'original imdb', 'angle separated imdb'}, 'Location','southeast');
  hold off

  subplot(1,2,2);
  hold on;
  bar(y_wo_relu);
  ylim([-0.1, 1.1]);
  errorbar(std_errors_x_location, std_errors_y_location_wo_relu, std_errors_value_wo_relu);
  legend({'original imdb', 'angle separated imdb'}, 'Location','southeast');
  hold off

  tmp_string = sprintf('1-KNN - %s', dataset);
  suptitle(tmp_string);
  saveas(h, fullfile(getDevPath(), 'temp_images', sprintf('%s.png', tmp_string)));


% -------------------------------------------------------------------------
function all_experiments_single_run = runAllExperimentsOnce(dataset, posneg_balance)
% -------------------------------------------------------------------------
  [~, experiments] = setupExperimentsUsingProjectedImbds(dataset, posneg_balance, 0);

  for i = 1 : numel(experiments)
    input_opts = {};
    input_opts.dataset = dataset;
    input_opts.imdb = experiments{i}.imdb;
    [~, experiments{i}.performance_summary] = testKnn(input_opts);
    % all_experiments_repeated{i}.performance_summary.testing.test.accuracy = []
  end

  for i = 1 : numel(experiments)
    afprintf(sprintf( ...
      '[INFO] 1-KNN Results for `%s`: \t\t train acc = %.4f, test acc = %.4f \n\n', ...
      experiments{i}.title, ...
      experiments{i}.performance_summary.testing.train.accuracy, ...
      experiments{i}.performance_summary.testing.test.accuracy));
  end
  all_experiments_single_run = experiments;
