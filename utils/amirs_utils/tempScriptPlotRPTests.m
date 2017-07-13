% -------------------------------------------------------------------------
function tempScriptPlotRPTests(all_experiments_multi_run, plot_title, save_results)
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

  y_all = [];
  y_wo_relu = [];
  y_w_relu = [];
  std_errors_value_all = [];
  std_errors_value_wo_relu = [];
  std_errors_value_w_relu = [];
  exp_number = 1;
  for j = 1:2
    for i = 1:11
      y_all(i,j) = mean(all_experiments_multi_run{exp_number}.performance);
      std_errors_value_all(i,j) = std(all_experiments_multi_run{exp_number}.performance);
      exp_number = exp_number + 1;
    end
  end

  y_wo_relu = [y_all(1,:); y_all(2:6,:)];
  y_w_relu = [y_all(1,:); y_all(7:11,:)];

  std_errors_value_wo_relu = reshape([std_errors_value_all(1,:); std_errors_value_all(2:6,:)]', 1, []);
  std_errors_value_w_relu = reshape([std_errors_value_all(1,:); std_errors_value_all(7:11,:)]', 1, []);

  std_errors_x_location = [ ...
    0.86, 1.14, ...
    1.86, 2.14, ...
    2.86, 3.14, ...
    3.86, 4.14, ...
    4.86, 5.14, ...
    5.86, 6.14];
  std_errors_y_location = reshape(y_all', 1, []);
  std_errors_y_location_wo_relu = cat(2, std_errors_y_location(1:2), std_errors_y_location(3:12));
  std_errors_y_location_w_relu = cat(2, std_errors_y_location(1:2), std_errors_y_location(13:22));

  h = figure;

  subplot(1,2,1);
  subplotBeef(y_wo_relu, std_errors_x_location, std_errors_y_location_wo_relu, std_errors_value_wo_relu, 'dense RP w/o ReLU');

  subplot(1,2,2);
  subplotBeef(y_w_relu, std_errors_x_location, std_errors_y_location_w_relu, std_errors_value_w_relu, 'dense RP w/ ReLU');

  % suptitle(plot_title);
  suptitle(plot_title);
  if save_results
    % saveas(h, fullfile(getDevPath(), 'temp_images', sprintf('%s.png', plot_title)));
    print(fullfile(getDevPath(), 'temp_images', plot_title), '-dpdf', '-fillpage')
  end

% -------------------------------------------------------------------------
function subplotBeef(y, std_errors_x_location, std_errors_y_location, std_errors_value, title_string)
% -------------------------------------------------------------------------
  hold on;
  bar(y);
  % ylim([-0.1, 1.1]);
  if max(y(:)) <= 1
    ylim([-0.1, 1]);
  else
    ylim([-0.1, inf]);
  end
  errorbar(std_errors_x_location, std_errors_y_location, std_errors_value);
  if isnan(y(1,2))
    legend({'original imdb'}, 'Location','southeast');
  else
    legend({'original imdb', 'angle separated imdb'}, 'Location','southeast');
  end
  title(title_string);

  % Set the X-Tick locations so that every other month is labeled.
  Xt = 1:1:6;
  Xl = [0.5 6.5];
  set(gca, 'XTick', Xt, 'XLim', Xl);

  % Add the months as tick labels.
  labels = ['Default';
            'RP =  1';
            'RP =  2';
            'RP =  3';
            'RP =  4';
            'RP =  5'];
  ax = axis;     % Current axis limits
  axis(axis);    % Set the axis limit modes (e.g. XLimMode) to manual
  Yl = ax(3:4);  % Y-axis limits

  % Place the text labels
  t = text(Xt, Yl(1) * ones(1, length(Xt)), labels);
  set(t, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'Rotation', 45);

  % Remove the default labels
  set(gca,'XTickLabel','');

  % Add values on the bars themselves
  values = reshape(y', 1, []);
  string_values_cell_array = {};
  for value = values
    if isnan(value)
      string_values_cell_array{end+1} = '-------';
    else
      if isinf(value)
        value = '9999999';
      end
      tmp = sprintf('%.5f', value);
      string_values_cell_array{end+1} = tmp(1:7); % HACK ALERT!: string width = 7 for later reshape
    end
  end

  string_values_matrix = reshape(cell2mat(string_values_cell_array)', 7, [])';
  t2 = text(std_errors_x_location - 0.035, values + 0.01, string_values_matrix);
  set(t2, 'HorizontalAlignment', 'Left', 'VerticalAlignment', 'middle', 'Rotation', 90);
  hold off














