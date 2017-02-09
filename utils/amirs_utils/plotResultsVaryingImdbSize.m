function plotResultsVaryingImdbSize()
  dataset = 'mnist-two-class-9-4';
  x = [38, 100, 266, 707, 1880]; %, 5000];
  y_single_larp   = [91.62, 92.88, 95.71, 97.64, 98.20]; %, 98.41];
  y_single_cnn    = [91.41, 86.09, 96.15, 97.62, 99.13]; %, 99.54];
  y_ensemble_larp = [95.81, 96.15, 97.14, 97.84, 97.74]; %, 97.42];
  y_ensemble_cnn  = [87.25, 95.76, 97.64, 98.61, 98.96]; %, 99.20];
  plotAndSaveShit(dataset, x, y_single_larp, y_single_cnn, y_ensemble_larp, y_ensemble_cnn);


  dataset = 'svhn-two-class-9-4';
  x = [38, 100, 266, 707, 1880]; %, 5000];
  y_single_larp   = [55.59, 63.48, 72.78, 83.42, 87.99]; %, 90.56];
  y_single_cnn    = [66.52, 81.41, 91.41, 93.66, 95.12]; %, 96.93];
  y_ensemble_larp = [65.85, 77.29, 79.84, 81.72, 81.61]; %, 81.68];
  y_ensemble_cnn  = [71.59, 82.75, 90.85, 92.74, 92.47]; %, 92.85];
  plotAndSaveShit(dataset, x, y_single_larp, y_single_cnn, y_ensemble_larp, y_ensemble_cnn);


  dataset = 'cifar-two-class-deer-truck';
  x = [38, 100, 266, 707, 1880]; %, 5000];
  y_single_larp   = [77.57, 82.97, 87.60, 90.27, 92.23]; %, 93.78];
  y_single_cnn    = [80.23, 84.85, 87.38, 90.15, 94.55]; %, 96.80];
  y_ensemble_larp = [81.37, 86.55, 89.57, 89.68, 90.05]; %, 91.20];
  y_ensemble_cnn  = [82.70, 85.63, 89.62, 92.17, 93.10]; %, 94.12];
  plotAndSaveShit(dataset, x, y_single_larp, y_single_cnn, y_ensemble_larp, y_ensemble_cnn);


function plotAndSaveShit(dataset, x, y_single_larp, y_single_cnn, y_ensemble_larp, y_ensemble_cnn)

  experiment_title = sprintf('%s - Ens. LaRP vs Single CNN for various balanced dataset sizes', dataset);
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
