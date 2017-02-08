function plotResultsVaryingImdbSize()
  dataset = 'mnist-two-class-9-4';
  x = [100, 266, 707, 1880];
  y_single_larp   = [92.88, 95.71, 97.64, 98.20];
  y_single_cnn    = [86.09, 96.15, 97.62, 99.13];
  y_ensemble_larp = [96.15, 97.14, 97.84, 97.74];
  y_ensemble_cnn  = [95.76, 97.64, 98.61, 98.96];
  plotAndSaveShit(dataset, x, y_single_larp, y_single_cnn, y_ensemble_larp, y_ensemble_cnn);


  dataset = 'svhn-two-class-9-4';
  x = [100, 266, 707, 1880];
  y_single_larp   = [63.48, 72.78, 83.42, 87.99];
  y_single_cnn    = [81.41, 91.41, 93.66, 95.12];
  y_ensemble_larp = [77.29, 79.84, 81.72, 81.61];
  y_ensemble_cnn  = [82.75, 90.85, 92.74, 92.47];
  plotAndSaveShit(dataset, x, y_single_larp, y_single_cnn, y_ensemble_larp, y_ensemble_cnn);


  dataset = 'cifar-two-class-deer-truck';
  x = [100, 266, 707, 1880];
  y_single_larp   = [82.97, 87.60, 90.27, 92.23];
  y_single_cnn    = [84.85, 87.38, 90.15, 94.55];
  y_ensemble_larp = [86.55, 89.57, 89.68, 90.05];
  y_ensemble_cnn  = [85.63, 89.62, 92.17, 93.10];
  plotAndSaveShit(dataset, x, y_single_larp, y_single_cnn, y_ensemble_larp, y_ensemble_cnn);


function plotAndSaveShit(dataset, x, y_single_larp, y_single_cnn, y_ensemble_larp, y_ensemble_cnn)

  experiment_title = sprintf('%s - Comparing Ens. LaRP and Single CNN for various balanced dataset sizes', dataset);
  h = figure;
  set(gca, 'fontsize', 12)
  hold on;
  grid on;

  semilogx( ...
    x, y_single_larp  , 'b', ...
    x, y_single_cnn   , 'k', ...
    x, y_ensemble_larp, 'b--', ...
    x, y_ensemble_cnn , 'k--', ...
    'LineWidth', 2);

  title(experiment_title);
  xlabel('Dataset Size')
  % ylim([0, 1]);
  % legend(legend_name_list);
  legend(...
    'Single LaRP', ...
    'Single CNN', ...
    'Ensemble LaRP', ...
    'Ensemble CNN');
  legend('Location','southeast');
  ylabel('Test Error');
  fileName = sprintf('%s_varying_balanced_dataset_size_test_comparison.eps', dataset);
  saveas(h, fileName, 'epsc');







