
% dummy run just to get fieldnames and initialize results arrays
fprintf('Dummy iteration...\t');
output = approximateKernelTestCode(false, 1, 'uci-ion');
results_per_fieldname_multirun = {};
results_per_fieldname_multidim = {};
all_fieldnames = fieldnames(output);
for i = 1 : numel(all_fieldnames)
  fieldname = all_fieldnames{i};
  results_per_fieldname_multirun.(fieldname) = [];
  results_per_fieldname_multidim.(fieldname).mean = [];
  results_per_fieldname_multidim.(fieldname).std = [];
end
fprintf('done.\n');





projected_dim_list = [1,2:4:22]; dataset = 'uci-ion';
% projected_dim_list = [1,2:4:34]; dataset = 'uci-ion';
% projected_dim_list = [1,5:10:55,60]; dataset = 'uci-sonar';
% projected_dim_list = 1:4;        dataset = 'uci-balance';
num_trials = 10;

for i = 1:numel(projected_dim_list)
  projected_dim = projected_dim_list(i);
  for i = 1 : num_trials
    fprintf('Iteration #%02d/%02d...\t', i, num_trials);
    output = approximateKernelTestCode(false, projected_dim, dataset);
    for i = 1 : numel(all_fieldnames)
      fieldname = all_fieldnames{i};
      results_per_fieldname_multirun.(fieldname)(end+1) = output.(fieldname);
    end
    fprintf('done.\n');
  end

  for i = 1 : numel(all_fieldnames)
    fieldname = all_fieldnames{i};
    fprintf('Average %s (k = %d): \t %.4f +/- %.4f\n', strrep(fieldname, '_', ' '), projected_dim, mean(results_per_fieldname_multirun.(fieldname)), std(results_per_fieldname_multirun.(fieldname)));
    results_per_fieldname_multidim.(fieldname).mean(end+1) = mean(results_per_fieldname_multirun.(fieldname));
    results_per_fieldname_multidim.(fieldname).std(end+1) = std(results_per_fieldname_multirun.(fieldname));
  end

end





figure,

subplot(1,2,1)
hold on;
legend_cell_array = {};
ciplot(results_per_fieldname_multidim.accuracy_spca_actual_eigen.mean - results_per_fieldname_multidim.accuracy_spca_actual_eigen.std, results_per_fieldname_multidim.accuracy_spca_actual_eigen.mean + results_per_fieldname_multidim.accuracy_spca_actual_eigen.std, projected_dim_list, 'm'); legend_cell_array = [legend_cell_array, 'accuracy spca actual eigen (std)'];
ciplot(results_per_fieldname_multidim.accuracy_spca_approx_eigen.mean - results_per_fieldname_multidim.accuracy_spca_approx_eigen.std, results_per_fieldname_multidim.accuracy_spca_approx_eigen.mean + results_per_fieldname_multidim.accuracy_spca_approx_eigen.std, projected_dim_list, 'y'); legend_cell_array = [legend_cell_array, 'accuracy spca approx eigen (std)'];
ciplot(results_per_fieldname_multidim.accuracy_kspca_actual_eigen.mean - results_per_fieldname_multidim.accuracy_kspca_actual_eigen.std, results_per_fieldname_multidim.accuracy_kspca_actual_eigen.mean + results_per_fieldname_multidim.accuracy_kspca_actual_eigen.std, projected_dim_list, 'r'); legend_cell_array = [legend_cell_array, 'accuracy kspca actual eigen (std)'];
ciplot(results_per_fieldname_multidim.accuracy_kspca_approx_eigen.mean - results_per_fieldname_multidim.accuracy_kspca_approx_eigen.std, results_per_fieldname_multidim.accuracy_kspca_approx_eigen.mean + results_per_fieldname_multidim.accuracy_kspca_approx_eigen.std, projected_dim_list, 'k'); legend_cell_array = [legend_cell_array, 'accuracy kspca approx eigen (std)'];
% ciplot(results_per_fieldname_multidim.accuracy_spca_approx_direct.mean - results_per_fieldname_multidim.accuracy_spca_approx_direct.std, results_per_fieldname_multidim.accuracy_spca_approx_direct.mean + results_per_fieldname_multidim.accuracy_spca_approx_direct.std, projected_dim_list, 'g'); legend_cell_array = [legend_cell_array, 'accuracy spca approx direct (std)'];
% ciplot(results_per_fieldname_multidim.accuracy_kspca_approx_direct.mean - results_per_fieldname_multidim.accuracy_kspca_approx_direct.std, results_per_fieldname_multidim.accuracy_kspca_approx_direct.mean + results_per_fieldname_multidim.accuracy_kspca_approx_direct.std, projected_dim_list, 'b'); legend_cell_array = [legend_cell_array, 'accuracy kspca approx direct (std)'];
% ciplot(results_per_fieldname_multidim.accuracy_kspca_actual_direct.mean - results_per_fieldname_multidim.accuracy_kspca_actual_direct.std, results_per_fieldname_multidim.accuracy_kspca_actual_direct.mean + results_per_fieldname_multidim.accuracy_kspca_actual_direct.std, projected_dim_list, 'c'); legend_cell_array = [legend_cell_array, 'accuracy kspca actual direct (std)'];
plot(projected_dim_list, results_per_fieldname_multidim.accuracy_spca_actual_eigen.mean, '--mo', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'accuracy spca actual eigen (mean)'];
plot(projected_dim_list, results_per_fieldname_multidim.accuracy_spca_approx_eigen.mean, '--yo', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'accuracy spca approx eigen (mean)'];
plot(projected_dim_list, results_per_fieldname_multidim.accuracy_kspca_actual_eigen.mean, '-r^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'accuracy kspca actual eigen (mean)'];
plot(projected_dim_list, results_per_fieldname_multidim.accuracy_kspca_approx_eigen.mean, '-k^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'accuracy kspca approx eigen (mean)'];
% plot(projected_dim_list, results_per_fieldname_multidim.accuracy_spca_approx_direct.mean, '--go', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'accuracy spca approx direct (mean)'];
% plot(projected_dim_list, results_per_fieldname_multidim.accuracy_kspca_approx_direct.mean, '-b^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'accuracy kspca approx direct (mean)'];
% plot(projected_dim_list, results_per_fieldname_multidim.accuracy_kspca_actual_direct.mean, '-c^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'accuracy kspca actual direct (mean)'];
xlabel('Projected Dimension');
ylabel('Accuracy (1-NN)');
hold off;
ylim([0,1]);
title('Accuracy Comparison');
legend(legend_cell_array, 'Location','southeast');


subplot(1,2,2)
hold on;
legend_cell_array = {};
ciplot(results_per_fieldname_multidim.duration_spca_actual_eigen.mean - results_per_fieldname_multidim.duration_spca_actual_eigen.std, results_per_fieldname_multidim.duration_spca_actual_eigen.mean + results_per_fieldname_multidim.duration_spca_actual_eigen.std, projected_dim_list, 'm'); legend_cell_array = [legend_cell_array, 'duration spca actual eigen (std)'];
ciplot(results_per_fieldname_multidim.duration_spca_approx_eigen.mean - results_per_fieldname_multidim.duration_spca_approx_eigen.std, results_per_fieldname_multidim.duration_spca_approx_eigen.mean + results_per_fieldname_multidim.duration_spca_approx_eigen.std, projected_dim_list, 'y'); legend_cell_array = [legend_cell_array, 'duration spca approx eigen (std)'];
ciplot(results_per_fieldname_multidim.duration_kspca_actual_eigen.mean - results_per_fieldname_multidim.duration_kspca_actual_eigen.std, results_per_fieldname_multidim.duration_kspca_actual_eigen.mean + results_per_fieldname_multidim.duration_kspca_actual_eigen.std, projected_dim_list, 'r'); legend_cell_array = [legend_cell_array, 'duration kspca actual eigen (std)'];
ciplot(results_per_fieldname_multidim.duration_kspca_approx_eigen.mean - results_per_fieldname_multidim.duration_kspca_approx_eigen.std, results_per_fieldname_multidim.duration_kspca_approx_eigen.mean + results_per_fieldname_multidim.duration_kspca_approx_eigen.std, projected_dim_list, 'k'); legend_cell_array = [legend_cell_array, 'duration kspca approx eigen (std)'];
% ciplot(results_per_fieldname_multidim.duration_spca_approx_direct.mean - results_per_fieldname_multidim.duration_spca_approx_direct.std, results_per_fieldname_multidim.duration_spca_approx_direct.mean + results_per_fieldname_multidim.duration_spca_approx_direct.std, projected_dim_list, 'g'); legend_cell_array = [legend_cell_array, 'duration spca approx direct (std)'];
% ciplot(results_per_fieldname_multidim.duration_kspca_approx_direct.mean - results_per_fieldname_multidim.duration_kspca_approx_direct.std, results_per_fieldname_multidim.duration_kspca_approx_direct.mean + results_per_fieldname_multidim.duration_kspca_approx_direct.std, projected_dim_list, 'b'); legend_cell_array = [legend_cell_array, 'duration kspca approx direct (std)'];
% ciplot(results_per_fieldname_multidim.duration_kspca_actual_direct.mean - results_per_fieldname_multidim.duration_kspca_actual_direct.std, results_per_fieldname_multidim.duration_kspca_actual_direct.mean + results_per_fieldname_multidim.duration_kspca_actual_direct.std, projected_dim_list, 'c'); legend_cell_array = [legend_cell_array, 'duration kspca actual direct (std)'];
plot(projected_dim_list, results_per_fieldname_multidim.duration_spca_actual_eigen.mean, '--mo', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'duration spca actual eigen (mean)'];
plot(projected_dim_list, results_per_fieldname_multidim.duration_spca_approx_eigen.mean, '--yo', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'duration spca approx eigen (mean)'];
plot(projected_dim_list, results_per_fieldname_multidim.duration_kspca_actual_eigen.mean, '-r^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'duration kspca actual eigen (mean)'];
plot(projected_dim_list, results_per_fieldname_multidim.duration_kspca_approx_eigen.mean, '-k^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'duration kspca approx eigen (mean)'];
% plot(projected_dim_list, results_per_fieldname_multidim.duration_spca_approx_direct.mean, '--go', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'duration spca approx direct (mean)'];
% plot(projected_dim_list, results_per_fieldname_multidim.duration_kspca_approx_direct.mean, '-b^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'duration kspca approx direct (mean)'];
% plot(projected_dim_list, results_per_fieldname_multidim.duration_kspca_actual_direct.mean, '-c^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'duration kspca actual direct (mean)'];
xlabel('Projected Dimension');
ylabel('Duration (sec)');
hold off;
title('Duration Comparison');
legend(legend_cell_array, 'Location','northwest');

suptitle(dataset)










% figure,
% hold on,
% x = 0:4;
% y_1 = [mean(results_per_fieldname_multirun.test_accuracy_rp_0), mean(results_per_fieldname_multirun.test_accuracy_rp_1), mean(results_per_fieldname_multirun.test_accuracy_rp_2), mean(results_per_fieldname_multirun.test_accuracy_rp_3), mean(results_per_fieldname_multirun.test_accuracy_rp_4)];
% y_2 = [mean(results_per_fieldname_multirun.test_accuracy_proposed_0), mean(results_per_fieldname_multirun.test_accuracy_proposed_1), mean(results_per_fieldname_multirun.test_accuracy_proposed_2), mean(results_per_fieldname_multirun.test_accuracy_proposed_3), mean(results_per_fieldname_multirun.test_accuracy_proposed_4)];
% plot(x, y_1, 'k--','LineWidth', 2)
% plot(x, y_2, 'b','LineWidth', 2)
% ylim([0.6,1])
% xlabel('Number of Projection Layers');
% ylabel('1-NN Classification Accuracy');
% legend({'Random Projections', 'Proposed Method (Psi H K)'})



























