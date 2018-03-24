% % [TODOs March 16]
% % 1) figure out a normalization scheme so that I can repro Elnaz's classification results.
% % 2) realize that SPCA-direct is in fact a scaled version of SPCA-eigen, so results should differ.
% % 3) realize that KSPCA-direct may not have anything to do with KSPCA-eigen as we still need to do the derivation for that.
% % 4) realize that a lot of bases are required for proper kernel approximation which results in poor time performance; overcome this by using FastFood instead of RKS.
% % 5) to really show the power of KSPCA direct, we should test on datasets with many samples (which would kill both the kernel construction time, and the
% %    SVD-decomposition)... currently, I've testing on usps-25-25 and uci-spam-250-100.. in both cases SPCA performs better than KSPCA.. where SPCA is already
% %    so much faster than KSPCA-{eigen, direct} and has better 1-NN. Therefore, we need to find a dataset (or maybe even test on regression instead of classification)
% %    where KSPCA-{eigen,direct} > SPCA in 1-NN so that we can argue the use of KSPCA-direct and its time superiority





% % projected_dim_list = [1,5:5:25,50:25:100]; dataset = 'usps';
% % projected_dim_list = [1,5:5:25,50]; dataset = 'uci-spam';
% % projected_dim_list = [1,2:8:34]; dataset = 'uci-ion';
% % projected_dim_list = [1,5:25:55,60]; dataset = 'uci-sonar';
% % projected_dim_list = 1:4;        dataset = 'uci-balance';
% projected_dim_list = [1,2:4:10];        dataset = 'xor-10D-350-train-150-test';
% projected_dim_list = [1,2:4:10];        dataset = 'rings-10D-350-train-150-test';
% projected_dim_list = [1,2:4:10];        dataset = 'spirals-10D-350-train-150-test';
% projected_dim_list = [2];        dataset = 'xor-10D-350-train-150-test';
num_trials = 10;






% % dummy run just to get fieldnames and initialize results arrays
% fprintf('Dummy iteration...\t');
% output = approximateKernelTestCode(false, 2, dataset);
% results_per_fieldname_multirun = {};
% results_per_fieldname_multidim = {};
% all_fieldnames = fieldnames(output);
% for i = 1 : numel(all_fieldnames)
%   fieldname = all_fieldnames{i};
%   results_per_fieldname_multirun.(fieldname) = [];
%   results_per_fieldname_multidim.(fieldname).mean = [];
%   results_per_fieldname_multidim.(fieldname).std = [];
% end
% fprintf('done.\n\n\n');


% for i = 1:numel(projected_dim_list)
%   projected_dim = projected_dim_list(i);
%   for i = 1 : num_trials
%     fprintf('Iteration #%02d/%02d...\t', i, num_trials);
%     output = approximateKernelTestCode(false, projected_dim, dataset);
%     for i = 1 : numel(all_fieldnames)
%       fieldname = all_fieldnames{i};
%       results_per_fieldname_multirun.(fieldname)(end+1) = output.(fieldname);
%     end
%     fprintf('done.\n');
%   end

%   for i = 1 : numel(all_fieldnames)
%     fieldname = all_fieldnames{i};
%     fprintf('Average %s (k = %d): \t %.4f +/- %.4f\n', strrep(fieldname, '_', ' '), projected_dim, mean(results_per_fieldname_multirun.(fieldname)), std(results_per_fieldname_multirun.(fieldname)));
%     results_per_fieldname_multidim.(fieldname).mean(end+1) = mean(results_per_fieldname_multirun.(fieldname));
%     results_per_fieldname_multidim.(fieldname).std(end+1) = std(results_per_fieldname_multirun.(fieldname));
%   end

% end





% figure,

% subplot(1,2,1)
% grid on;
% hold on;
% legend_cell_array = {};
% % ciplot(results_per_fieldname_multidim.accuracy_spca_actual_eigen.mean - results_per_fieldname_multidim.accuracy_spca_actual_eigen.std, results_per_fieldname_multidim.accuracy_spca_actual_eigen.mean + results_per_fieldname_multidim.accuracy_spca_actual_eigen.std, projected_dim_list, 'm'); legend_cell_array = [legend_cell_array, 'accuracy spca actual eigen (std)'];
% % ciplot(results_per_fieldname_multidim.accuracy_kspca_actual_eigen.mean - results_per_fieldname_multidim.accuracy_kspca_actual_eigen.std, results_per_fieldname_multidim.accuracy_kspca_actual_eigen.mean + results_per_fieldname_multidim.accuracy_kspca_actual_eigen.std, projected_dim_list, 'r'); legend_cell_array = [legend_cell_array, 'accuracy kspca actual eigen (std)'];
% % ciplot(results_per_fieldname_multidim.accuracy_spca_approx_direct.mean - results_per_fieldname_multidim.accuracy_spca_approx_direct.std, results_per_fieldname_multidim.accuracy_spca_approx_direct.mean + results_per_fieldname_multidim.accuracy_spca_approx_direct.std, projected_dim_list, 'g'); legend_cell_array = [legend_cell_array, 'accuracy spca approx direct (std)'];
% % ciplot(results_per_fieldname_multidim.accuracy_kspca_approx_direct.mean - results_per_fieldname_multidim.accuracy_kspca_approx_direct.std, results_per_fieldname_multidim.accuracy_kspca_approx_direct.mean + results_per_fieldname_multidim.accuracy_kspca_approx_direct.std, projected_dim_list, 'b'); legend_cell_array = [legend_cell_array, 'accuracy kspca approx direct (std)'];
% plot(projected_dim_list, results_per_fieldname_multidim.accuracy_spca_actual_eigen.mean, '--mo', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'accuracy spca actual eigen (mean)'];
% plot(projected_dim_list, results_per_fieldname_multidim.accuracy_kspca_actual_eigen.mean, '-r^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'accuracy kspca actual eigen (mean)'];
% plot(projected_dim_list, results_per_fieldname_multidim.accuracy_spca_approx_direct.mean, '--go', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'accuracy spca approx direct (mean)'];
% plot(projected_dim_list, results_per_fieldname_multidim.accuracy_kspca_approx_direct.mean, '-b^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'accuracy kspca approx direct (mean)'];
% xlabel('Projected Dimension');
% ylabel('Accuracy (1-NN)');
% hold off;
% ylim([0,1]);
% title('Accuracy Comparison');
% legend(legend_cell_array, 'Location','southeast');


% subplot(1,2,2)
% grid on;
% hold on;
% legend_cell_array = {};
% ciplot(results_per_fieldname_multidim.duration_spca_actual_eigen.mean - results_per_fieldname_multidim.duration_spca_actual_eigen.std, results_per_fieldname_multidim.duration_spca_actual_eigen.mean + results_per_fieldname_multidim.duration_spca_actual_eigen.std, projected_dim_list, 'm'); legend_cell_array = [legend_cell_array, 'duration spca actual eigen (std)'];
% ciplot(results_per_fieldname_multidim.duration_kspca_actual_eigen.mean - results_per_fieldname_multidim.duration_kspca_actual_eigen.std, results_per_fieldname_multidim.duration_kspca_actual_eigen.mean + results_per_fieldname_multidim.duration_kspca_actual_eigen.std, projected_dim_list, 'r'); legend_cell_array = [legend_cell_array, 'duration kspca actual eigen (std)'];
% ciplot(results_per_fieldname_multidim.duration_spca_approx_direct.mean - results_per_fieldname_multidim.duration_spca_approx_direct.std, results_per_fieldname_multidim.duration_spca_approx_direct.mean + results_per_fieldname_multidim.duration_spca_approx_direct.std, projected_dim_list, 'g'); legend_cell_array = [legend_cell_array, 'duration spca approx direct (std)'];
% ciplot(results_per_fieldname_multidim.duration_kspca_approx_direct.mean - results_per_fieldname_multidim.duration_kspca_approx_direct.std, results_per_fieldname_multidim.duration_kspca_approx_direct.mean + results_per_fieldname_multidim.duration_kspca_approx_direct.std, projected_dim_list, 'b'); legend_cell_array = [legend_cell_array, 'duration kspca approx direct (std)'];
% plot(projected_dim_list, results_per_fieldname_multidim.duration_spca_actual_eigen.mean, '--mo', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'duration spca actual eigen (mean)'];
% plot(projected_dim_list, results_per_fieldname_multidim.duration_kspca_actual_eigen.mean, '-r^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'duration kspca actual eigen (mean)'];
% plot(projected_dim_list, results_per_fieldname_multidim.duration_spca_approx_direct.mean, '--go', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'duration spca approx direct (mean)'];
% plot(projected_dim_list, results_per_fieldname_multidim.duration_kspca_approx_direct.mean, '-b^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'duration kspca approx direct (mean)'];
% xlabel('Projected Dimension');
% ylabel('Duration (sec)');
% hold off;
% title('Duration Comparison');
% legend(legend_cell_array, 'Location','northwest');

% suptitle(dataset)


























































num_trials = 40;
% dataset = 'xor-10D-350-train-150-test';
% dataset = 'rings-10D-350-train-150-test';
dataset = 'spirals-10D-350-train-150-test';
% dataset = 'usps';
% dataset = 'uci-sonar';
% dataset = 'uci-spam';
% dataset = 'uci-ion';

% dummy run just to get fieldnames and initialize results arrays
fprintf('Dummy iteration...\t');
output = approximateKernelTestCode(false, 2, dataset);
results_per_fieldname_multirun = {};
all_fieldnames = fieldnames(output);
for i = 1 : numel(all_fieldnames)
  fieldname = all_fieldnames{i};
  results_per_fieldname_multirun.(fieldname) = [];
end
fprintf('done.\n\n\n');


for i = 1 : num_trials
  fprintf('Iteration #%02d/%02d...\t', i, num_trials);
  output = approximateKernelTestCode(false, -1, dataset);
  for i = 1 : numel(all_fieldnames)
    fieldname = all_fieldnames{i};
    results_per_fieldname_multirun.(fieldname)(end+1) = output.(fieldname);
  end
  fprintf('done.\n');
end


proposed_method.mean = [mean(results_per_fieldname_multirun.test_accuracy_proposed_0), mean(results_per_fieldname_multirun.test_accuracy_proposed_1), mean(results_per_fieldname_multirun.test_accuracy_proposed_2), mean(results_per_fieldname_multirun.test_accuracy_proposed_3), mean(results_per_fieldname_multirun.test_accuracy_proposed_4)];
proposed_method.std = [std(results_per_fieldname_multirun.test_accuracy_proposed_0), std(results_per_fieldname_multirun.test_accuracy_proposed_1), std(results_per_fieldname_multirun.test_accuracy_proposed_2), std(results_per_fieldname_multirun.test_accuracy_proposed_3), std(results_per_fieldname_multirun.test_accuracy_proposed_4)];
random_p_method.mean = [mean(results_per_fieldname_multirun.test_accuracy_rp_0), mean(results_per_fieldname_multirun.test_accuracy_rp_1), mean(results_per_fieldname_multirun.test_accuracy_rp_2), mean(results_per_fieldname_multirun.test_accuracy_rp_3), mean(results_per_fieldname_multirun.test_accuracy_rp_4)];
random_p_method.std = [std(results_per_fieldname_multirun.test_accuracy_rp_0), std(results_per_fieldname_multirun.test_accuracy_rp_1), std(results_per_fieldname_multirun.test_accuracy_rp_2), std(results_per_fieldname_multirun.test_accuracy_rp_3), std(results_per_fieldname_multirun.test_accuracy_rp_4)];
num_layers_list = 1 : numel(proposed_method.mean);

figure,
grid on;
hold on;
legend_cell_array = {};
ciplot(proposed_method.mean - proposed_method.std, proposed_method.mean + proposed_method.std, num_layers_list, 'r'); legend_cell_array = [legend_cell_array, 'proposed method (std)'];
ciplot(random_p_method.mean - random_p_method.std, random_p_method.mean + random_p_method.std, num_layers_list, 'b'); legend_cell_array = [legend_cell_array, 'random projection method (std)'];

plot(num_layers_list, proposed_method.mean, '-r^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'accuracy proposed method (mean)'];
plot(num_layers_list, random_p_method.mean, '-b^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'accuracy random projection method (mean)'];

xlabel('# of Random Layers');
ylabel('Accuracy (1-NN)');
ylim([0,1.1]);
hold off;
title(sprintf('%s Accuracy Comparison', dataset));
legend(legend_cell_array, 'Location','southwest');



