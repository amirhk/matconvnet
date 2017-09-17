max_m_normals = @(m,x) m * normpdf(x) .* normcdf(x).^(m-1);

x = -5:0.01:5;
y_max_1 = max_m_normals(1, x);
y_max_2 = max_m_normals(2, x);
y_max_4 = max_m_normals(4, x);
y_max_8 = max_m_normals(8, x);
y_max_16 = max_m_normals(16, x);

y_max_1 = y_max_1 / sum(y_max_1); % normalize
y_max_2 = y_max_2 / sum(y_max_2); % normalize
y_max_4 = y_max_4 / sum(y_max_4); % normalize
y_max_8 = y_max_8 / sum(y_max_8); % normalize
y_max_16 = y_max_16 / sum(y_max_16); % normalize

figure,
hold on,
plot(x, y_max_1, 'LineWidth', 2);
plot(x, y_max_2, 'LineWidth', 2);
plot(x, y_max_4, 'LineWidth', 2);
plot(x, y_max_8, 'LineWidth', 2);
plot(x, y_max_16, 'LineWidth', 2);
hold off

legend({ ...
  'max of 1 identical Gaussians',  ...
  'max of 2 identical Gaussians',  ...
  'max of 4 identical Gaussians',  ...
  'max of 8 identical Gaussians',  ...
  'max of 16 identical Gaussians', ...
  }, 'Location', 'northwest')


% sum(y_max_2 .* x) / (length(y_max_2) - 1)
% sum(y_max_2 .* x.^2) / (length(y_max_2) - 1)

% mean(x .* y_max_2)
% mean((x .* y_max_2).^2)

export_fig 'plot_max_m_normals' -eps


fprintf('[INFO] MAX of 1 - \t  Emperical mean: %.5f, Theoretical mean: %.5f \n', sum(y_max_1 .* x), 0);
fprintf('[INFO] MAX of 2 - \t  Emperical mean: %.5f, Theoretical mean: %.5f \n', sum(y_max_2 .* x), 1 / sqrt(pi) * 1);
fprintf('[INFO] MAX of 4 - \t  Emperical mean: %.5f, Theoretical mean: %.5f \n', sum(y_max_4 .* x), 1 / sqrt(pi) * 2);
fprintf('[INFO] MAX of 8 - \t  Emperical mean: %.5f, Theoretical mean: %.5f \n', sum(y_max_8 .* x), 1 / sqrt(pi) * (2 + sqrt(1 + 3 / pi)));

