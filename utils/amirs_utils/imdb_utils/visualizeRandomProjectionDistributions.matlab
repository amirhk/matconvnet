data_vector = 1:10;

repeat_count = 10000;

means = [];
vars = [];
color_palette = {'b', 'r', 'g', 'c', 'y'};
figure,
hold on,
grid on,
x_ticks = -100:2.5:100;
y_limits = [0 200];
i = 1;
legend_entries = {};
% for max_count = [1,10,100,1000,10000]
for max_count = [1,10,100]
  projected_coefficients = [];
  for k = 1 : repeat_count
    projected_coefficients(end + 1) = max(randn(max_count, length(data_vector)) * data_vector');
  end
  data = projected_coefficients;
  histogram( ...
    data, ...
    x_ticks, ...
    'facecolor', ...
    color_palette{mod(i - 1,numel(color_palette)) + 1}, ...
    'facealpha', ...
    0.4);
  means(end+1) = mean(projected_coefficients);
  vars(end+1) = var(projected_coefficients);
  legend_entries{end+1} = sprintf('max over %d', max_count);
  i = i + 1;
end
% ylim(y_limits);
legend(legend_entries, 'Location', 'NorthWest');
hold off
means
vars
