% [TODOs March 16]
% 1) figure out a normalization scheme so that I can repro Elnaz's classification results.
% 2) realize that SPCA-direct is in fact a scaled version of SPCA-eigen, so results should differ.
% 3) realize that KSPCA-direct may not have anything to do with KSPCA-eigen as we still need to do the derivation for that.
% 4) realize that a lot of bases are required for proper kernel approximation which results in poor time performance; overcome this by using FastFood instead of RKS.
% 5) to really show the power of KSPCA direct, we should test on datasets with many samples (which would kill both the kernel construction time, and the
%    SVD-decomposition)... currently, I've testing on usps-25-25 and uci-spam-250-100.. in both cases SPCA performs better than KSPCA.. where SPCA is already
%    so much faster than KSPCA-{eigen, direct} and has better 1-NN. Therefore, we need to find a dataset (or maybe even test on regression instead of classification)
%    where KSPCA-{eigen,direct} > SPCA in 1-NN so that we can argue the use of KSPCA-direct and its time superiority





% projected_dim_list = [1:5,5:5:25,50:25:100]; dataset = 'usps';
% projected_dim_list = [1:10]; dataset = 'usps';
% projected_dim_list = [1:2:9,10:5:25,25:25:100]; dataset = 'mnist-784';
% projected_dim_list = [25:25:100]; dataset = 'mnist-784';
% projected_dim_list = [1:10]; dataset = 'mnist-784';
projected_dim_list = [1:5,10:5:25,50:25:100]; dataset = 'imagenet-tiny';
% projected_dim_list = [1:2:9,10:5:25,25:25:100,100:100:700]; dataset = 'mnist-784';
% projected_dim_list = [1,5:5:25,50]; dataset = 'uci-spam';
% projected_dim_list = [1,2:8:34]; dataset = 'uci-ion';
% projected_dim_list = [1,2:2:34]; dataset = 'uci-ion';
% projected_dim_list = [1:2:34]; dataset = 'uci-ion';
% projected_dim_list = [1,5:10:55,60]; dataset = 'uci-sonar';
% projected_dim_list = 1:4;        dataset = 'uci-balance';
% projected_dim_list = [1,2:2:10];        dataset = 'xor-10D-350-train-150-test';
% projected_dim_list = [1,2:4:10];        dataset = 'rings-10D-350-train-150-test';
% projected_dim_list = [1,2:4:10];        dataset = 'spirals-10D-350-train-150-test';
num_trials = 10;






% dummy run just to get fieldnames and initialize results arrays
fprintf('Dummy iteration...\t');
output = approximateKernelTestCode(false, 2, dataset);
results_per_fieldname_singledim_multirun = {};
results_per_fieldname_multidim = {};
all_fieldnames = fieldnames(output);
for i = 1 : numel(all_fieldnames)
  fieldname = all_fieldnames{i};
  results_per_fieldname_multidim.(fieldname).mean = [];
  results_per_fieldname_multidim.(fieldname).std = [];
end
fprintf('done.\n\n\n');


for i = 1:numel(projected_dim_list)

  projected_dim = projected_dim_list(i);

  for i = 1 : numel(all_fieldnames)
    fieldname = all_fieldnames{i};
    results_per_fieldname_singledim_multirun.(fieldname) = [];
  end

  for i = 1 : num_trials
    fprintf('Iteration #%02d/%02d...\t', i, num_trials);
    output = approximateKernelTestCode(false, projected_dim, dataset);
    for i = 1 : numel(all_fieldnames)
      fieldname = all_fieldnames{i};
      results_per_fieldname_singledim_multirun.(fieldname)(end+1) = output.(fieldname);
    end
    fprintf('done.\n');
  end

  for i = 1 : numel(all_fieldnames)
    fieldname = all_fieldnames{i};
    fprintf('Average %s (k = %d): \t %.4f +/- %.4f\n', strrep(fieldname, '_', ' '), projected_dim, mean(results_per_fieldname_singledim_multirun.(fieldname)), std(results_per_fieldname_singledim_multirun.(fieldname)));
    results_per_fieldname_multidim.(fieldname).mean(end+1) = mean(results_per_fieldname_singledim_multirun.(fieldname));
    results_per_fieldname_multidim.(fieldname).std(end+1) = std(results_per_fieldname_singledim_multirun.(fieldname));
  end

end


% save(sprintf('%s', dataset), 'results_per_fieldname_multidim')

% keyboard

% % projected_dim_list = [1:2:9,10:5:25,25:25:100]; dataset = 'mnist-784';
% % projected_dim_list = [1,2:4:34]; dataset = 'uci-ion';
% projected_dim_list = [1:10];        dataset = 'xor-10D-350-train-150-test';

figure,

subplot(1,2,1)
grid on;
hold on;
legend_cell_array = {};
if isfield(results_per_fieldname_multidim, 'accuracy_spca_eigen'), plot(projected_dim_list, results_per_fieldname_multidim.accuracy_spca_eigen.mean, '--ro', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'Accuracy SPCA (mean)']; end
if isfield(results_per_fieldname_multidim, 'accuracy_spca_aeigen'), plot(projected_dim_list, results_per_fieldname_multidim.accuracy_spca_aeigen.mean, '--mo', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'Accuracy SPCA-x (mean)']; end
if isfield(results_per_fieldname_multidim, 'accuracy_spca_direct'), plot(projected_dim_list, results_per_fieldname_multidim.accuracy_spca_direct.mean, '--go', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'Accuracy SRP (mean)']; end
if isfield(results_per_fieldname_multidim, 'accuracy_kspca_eigen'), plot(projected_dim_list, results_per_fieldname_multidim.accuracy_kspca_eigen.mean, '-r^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'Accuracy KSPCA (mean)']; end
if isfield(results_per_fieldname_multidim, 'accuracy_kspca_aeigen'), plot(projected_dim_list, results_per_fieldname_multidim.accuracy_kspca_aeigen.mean, '-m^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'Accuracy KSPCA-x (mean)']; end
if isfield(results_per_fieldname_multidim, 'accuracy_kspca_direct'), plot(projected_dim_list, results_per_fieldname_multidim.accuracy_kspca_direct.mean, '-g^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'Accuracy KSRP (mean)']; end
if isfield(results_per_fieldname_multidim, 'accuracy_pca_direct'), plot(projected_dim_list, results_per_fieldname_multidim.accuracy_pca_direct.mean, '-.yo', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'Accuracy RPCA (mean)']; end
if isfield(results_per_fieldname_multidim, 'accuracy_random_projection'), plot(projected_dim_list, results_per_fieldname_multidim.accuracy_random_projection.mean, '-.c^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'Accuracy Random Projection (mean)']; end
xlabel('Projected Dimension', 'FontSize', 16);
ylabel('Accuracy (1-NN)', 'FontSize', 16);
hold off;
ylim([0,1]);
title('Accuracy Comparison', 'FontSize', 16);
legend(legend_cell_array, 'Location', 'east', 'FontSize', 12);

% saveas(gcf,sprintf('1nn_perf_%s', dataset),'epsc')


% figure
subplot(1,2,2)
grid on;
hold on;
legend_cell_array = {};
% ciplot(results_per_fieldname_multidim.duration_spca_eigen.mean - results_per_fieldname_multidim.duration_spca_eigen.std, results_per_fieldname_multidim.duration_spca_eigen.mean + results_per_fieldname_multidim.duration_spca_eigen.std, projected_dim_list, 'm'); legend_cell_array = [legend_cell_array, 'duration spca eigen (std)'];
% ciplot(results_per_fieldname_multidim.duration_kspca_eigen.mean - results_per_fieldname_multidim.duration_kspca_eigen.std, results_per_fieldname_multidim.duration_kspca_eigen.mean + results_per_fieldname_multidim.duration_kspca_eigen.std, projected_dim_list, 'r'); legend_cell_array = [legend_cell_array, 'duration kspca eigen (std)'];
% ciplot(results_per_fieldname_multidim.duration_spca_direct.mean - results_per_fieldname_multidim.duration_spca_direct.std, results_per_fieldname_multidim.duration_spca_direct.mean + results_per_fieldname_multidim.duration_spca_direct.std, projected_dim_list, 'g'); legend_cell_array = [legend_cell_array, 'duration spca direct (std)'];
% ciplot(results_per_fieldname_multidim.duration_kspca_direct.mean - results_per_fieldname_multidim.duration_kspca_direct.std, results_per_fieldname_multidim.duration_kspca_direct.mean + results_per_fieldname_multidim.duration_kspca_direct.std, projected_dim_list, 'b'); legend_cell_array = [legend_cell_array, 'duration kspca direct (std)'];
% ciplot(results_per_fieldname_multidim.duration_pca_direct.mean - results_per_fieldname_multidim.duration_pca_direct.std, results_per_fieldname_multidim.duration_pca_direct.mean + results_per_fieldname_multidim.duration_pca_direct.std, projected_dim_list, 'y'); legend_cell_array = [legend_cell_array, 'duration pca direct (std)'];
% ciplot(results_per_fieldname_multidim.duration_random_projection.mean - results_per_fieldname_multidim.duration_random_projection.std, results_per_fieldname_multidim.duration_random_projection.mean + results_per_fieldname_multidim.duration_random_projection.std, projected_dim_list, 'c'); legend_cell_array = [legend_cell_array, 'duration random projection (std)'];
if isfield(results_per_fieldname_multidim, 'duration_spca_eigen'), plot(projected_dim_list, results_per_fieldname_multidim.duration_spca_eigen.mean, '--ro', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'Duration SPCA (mean)']; end
if isfield(results_per_fieldname_multidim, 'duration_spca_aeigen'), plot(projected_dim_list, results_per_fieldname_multidim.duration_spca_aeigen.mean, '--mo', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'Duration SPCA-x (mean)']; end
if isfield(results_per_fieldname_multidim, 'duration_spca_direct'), plot(projected_dim_list, results_per_fieldname_multidim.duration_spca_direct.mean, '--go', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'Duration SRP (mean)']; end
if isfield(results_per_fieldname_multidim, 'duration_kspca_eigen'), plot(projected_dim_list, results_per_fieldname_multidim.duration_kspca_eigen.mean, '-r^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'Duration KSPCA (mean)']; end
if isfield(results_per_fieldname_multidim, 'duration_kspca_aeigen'), plot(projected_dim_list, results_per_fieldname_multidim.duration_kspca_aeigen.mean, '-m^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'Duration KSPCA-x (mean)']; end
if isfield(results_per_fieldname_multidim, 'duration_kspca_direct'), plot(projected_dim_list, results_per_fieldname_multidim.duration_kspca_direct.mean, '-g^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'Duration KSRP (mean)']; end
if isfield(results_per_fieldname_multidim, 'duration_pca_direct'), plot(projected_dim_list, results_per_fieldname_multidim.duration_pca_direct.mean, '-y^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'Duration RPCA (mean)']; end
if isfield(results_per_fieldname_multidim, 'duration_random_projection'), plot(projected_dim_list, results_per_fieldname_multidim.duration_random_projection.mean, '-c^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'Duration Random Projection (mean)']; end
xlabel('Projected Dimension', 'FontSize', 16);
ylabel('Duration (sec)', 'FontSize', 16);
hold off;
title('Duration Comparison', 'FontSize', 16);
legend(legend_cell_array, 'Location', 'west', 'FontSize', 12);


% disp(' -- -- -- -- -- -- -- ')
% results_per_fieldname_multidim.rank_spca_eigen_X.mean
% results_per_fieldname_multidim.rank_spca_eigen_X_test.mean
% disp(' -- -- -- -- -- -- -- ')
% results_per_fieldname_multidim.rank_spca_aeigen_X.mean
% results_per_fieldname_multidim.rank_spca_aeigen_X_test.mean
% disp(' -- -- -- -- -- -- -- ')
% results_per_fieldname_multidim.rank_spca_direct_X.mean
% results_per_fieldname_multidim.rank_spca_direct_X_test.mean
% disp(' -- -- -- -- -- -- -- ')
% results_per_fieldname_multidim.rank_kspca_eigen_X.mean
% results_per_fieldname_multidim.rank_kspca_eigen_X_test.mean
% disp(' -- -- -- -- -- -- -- ')
% results_per_fieldname_multidim.rank_kspca_aeigen_X.mean
% results_per_fieldname_multidim.rank_kspca_aeigen_X_test.mean
% disp(' -- -- -- -- -- -- -- ')
% results_per_fieldname_multidim.rank_kspca_direct_X.mean
% results_per_fieldname_multidim.rank_kspca_direct_X_test.mean
% disp(' -- -- -- -- -- -- -- ')

% saveas(gcf,sprintf('time_perf_%s', dataset),'epsc')

% keyboard

% h = suptitle(dataset);
% set(h,'FontSize',20,'FontWeight','normal');


% keyboard

% for i = 1 : numel(all_fieldnames)
%   fieldname = all_fieldnames{i};
%   fprintf('\n%s\n', fieldname);
%   for j = 1:numel(projected_dim_list)
%     fprintf('[d = %d] \t [mean] %.4f \t [std] %.4f\n', projected_dim_list(j), results_per_fieldname_multidim.(fieldname).mean(j), results_per_fieldname_multidim.(fieldname).std(j));
%   end
% end

























































% num_trials = 1;
% projected_dim = 32;
% dataset = 'xor-10D-350-train-150-test';
% % dataset = 'rings-10D-350-train-150-test';
% % dataset = 'spirals-10D-350-train-150-test';
% % dataset = 'mnist-784';
% % dataset = 'usps';
% % dataset = 'uci-sonar';
% % dataset = 'uci-spam';
% % dataset = 'uci-ion';
% % dataset = 'uci-balance';

% % dummy run just to get fieldnames and initialize results arrays
% fprintf('Dummy iteration...\n');
% output = approximateKernelTestCode(false, 2, dataset);
% results_per_fieldname_singledim_multirun = {};
% all_fieldnames = fieldnames(output);
% for i = 1 : numel(all_fieldnames)
%   fieldname = all_fieldnames{i};
%   results_per_fieldname_singledim_multirun.(fieldname) = [];
% end
% fprintf('done.\n\n\n');


% for i = 1 : num_trials
%   fprintf('Iteration #%02d/%02d...\n', i, num_trials);
%   output = approximateKernelTestCode(false, projected_dim, dataset);
%   for i = 1 : numel(all_fieldnames)
%     fieldname = all_fieldnames{i};
%     results_per_fieldname_singledim_multirun.(fieldname)(end+1) = output.(fieldname);
%   end
%   fprintf('done.\n');
% end


% proposed_method.accuracy.mean = [mean(results_per_fieldname_singledim_multirun.accuracy_proposed_0), mean(results_per_fieldname_singledim_multirun.accuracy_proposed_1), mean(results_per_fieldname_singledim_multirun.accuracy_proposed_2)]; % , mean(results_per_fieldname_singledim_multirun.accuracy_proposed_3), mean(results_per_fieldname_singledim_multirun.accuracy_proposed_4)];
% proposed_method.accuracy.std = [std(results_per_fieldname_singledim_multirun.accuracy_proposed_0), std(results_per_fieldname_singledim_multirun.accuracy_proposed_1), std(results_per_fieldname_singledim_multirun.accuracy_proposed_2)]; % , std(results_per_fieldname_singledim_multirun.accuracy_proposed_3), std(results_per_fieldname_singledim_multirun.accuracy_proposed_4)];
% proposed_method.duration.mean = [mean(results_per_fieldname_singledim_multirun.duration_proposed_0), mean(results_per_fieldname_singledim_multirun.duration_proposed_1), mean(results_per_fieldname_singledim_multirun.duration_proposed_2)]; % , mean(results_per_fieldname_singledim_multirun.duration_proposed_3), mean(results_per_fieldname_singledim_multirun.duration_proposed_4)];
% proposed_method.duration.std = [std(results_per_fieldname_singledim_multirun.duration_proposed_0), std(results_per_fieldname_singledim_multirun.duration_proposed_1), std(results_per_fieldname_singledim_multirun.duration_proposed_2)]; % , std(results_per_fieldname_singledim_multirun.duration_proposed_3), std(results_per_fieldname_singledim_multirun.duration_proposed_4)];

% backprop_method.accuracy.mean = [mean(results_per_fieldname_singledim_multirun.accuracy_backprop_0), mean(results_per_fieldname_singledim_multirun.accuracy_backprop_1), mean(results_per_fieldname_singledim_multirun.accuracy_backprop_2)]; % , mean(results_per_fieldname_singledim_multirun.accuracy_backprop_3), mean(results_per_fieldname_singledim_multirun.accuracy_backprop_4)];
% backprop_method.accuracy.std = [std(results_per_fieldname_singledim_multirun.accuracy_backprop_0), std(results_per_fieldname_singledim_multirun.accuracy_backprop_1), std(results_per_fieldname_singledim_multirun.accuracy_backprop_2)]; % , std(results_per_fieldname_singledim_multirun.accuracy_backprop_3), std(results_per_fieldname_singledim_multirun.accuracy_backprop_4)];
% backprop_method.duration.mean = [mean(results_per_fieldname_singledim_multirun.duration_backprop_0), mean(results_per_fieldname_singledim_multirun.duration_backprop_1), mean(results_per_fieldname_singledim_multirun.duration_backprop_2)]; % , mean(results_per_fieldname_singledim_multirun.duration_backprop_3), mean(results_per_fieldname_singledim_multirun.duration_backprop_4)];
% backprop_method.duration.std = [std(results_per_fieldname_singledim_multirun.duration_backprop_0), std(results_per_fieldname_singledim_multirun.duration_backprop_1), std(results_per_fieldname_singledim_multirun.duration_backprop_2)]; % , std(results_per_fieldname_singledim_multirun.duration_backprop_3), std(results_per_fieldname_singledim_multirun.duration_backprop_4)];

% % random_p_method.accuracy.mean = [mean(results_per_fieldname_singledim_multirun.accuracy_rp_0), mean(results_per_fieldname_singledim_multirun.accuracy_rp_1), mean(results_per_fieldname_singledim_multirun.accuracy_rp_2)]; % , mean(results_per_fieldname_singledim_multirun.accuracy_rp_3), mean(results_per_fieldname_singledim_multirun.accuracy_rp_4)];
% % random_p_method.accuracy.std = [std(results_per_fieldname_singledim_multirun.accuracy_rp_0), std(results_per_fieldname_singledim_multirun.accuracy_rp_1), std(results_per_fieldname_singledim_multirun.accuracy_rp_2)]; % , std(results_per_fieldname_singledim_multirun.accuracy_rp_3), std(results_per_fieldname_singledim_multirun.accuracy_rp_4)];
% % random_p_method.duration.mean = [mean(results_per_fieldname_singledim_multirun.duration_rp_0), mean(results_per_fieldname_singledim_multirun.duration_rp_1), mean(results_per_fieldname_singledim_multirun.duration_rp_2)]; % , mean(results_per_fieldname_singledim_multirun.duration_rp_3), mean(results_per_fieldname_singledim_multirun.duration_rp_4)];
% % random_p_method.duration.std = [std(results_per_fieldname_singledim_multirun.duration_rp_0), std(results_per_fieldname_singledim_multirun.duration_rp_1), std(results_per_fieldname_singledim_multirun.duration_rp_2)]; % , std(results_per_fieldname_singledim_multirun.duration_rp_3), std(results_per_fieldname_singledim_multirun.duration_rp_4)];

% num_layers_list = 1 : numel(proposed_method.accuracy.mean);

% figure,



% subplot(1,2,1)
% grid on;
% hold on;
% legend_cell_array = {};

% ciplot(proposed_method.accuracy.mean - proposed_method.accuracy.std, proposed_method.accuracy.mean + proposed_method.accuracy.std, num_layers_list, 'r'); legend_cell_array = [legend_cell_array, 'proposed method (std)'];
% ciplot(backprop_method.accuracy.mean - backprop_method.accuracy.std, backprop_method.accuracy.mean + backprop_method.accuracy.std, num_layers_list, 'b'); legend_cell_array = [legend_cell_array, 'backprop method (std)'];
% % ciplot(random_p_method.accuracy.mean - random_p_method.accuracy.std, random_p_method.accuracy.mean + random_p_method.accuracy.std, num_layers_list, 'b'); legend_cell_array = [legend_cell_array, 'random projection method (std)'];
% plot(num_layers_list, proposed_method.accuracy.mean, '-r^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'accuracy proposed method (mean)'];
% plot(num_layers_list, backprop_method.accuracy.mean, '-b^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'accuracy backprop method (mean)'];
% % plot(num_layers_list, random_p_method.accuracy.mean, '-b^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'accuracy random projection method (mean)'];

% xticks(1 : numel(proposed_method.accuracy.mean));
% xlabel('# of Random Layers');
% ylabel('Accuracy (1-NN)');
% ylim([0,1.1]);
% hold off;
% title('Accuracy Comparison');
% legend(legend_cell_array, 'Location','southwest');





% subplot(1,2,2)
% grid on;
% hold on;
% legend_cell_array = {};

% ciplot(proposed_method.duration.mean - proposed_method.duration.std, proposed_method.duration.mean + proposed_method.duration.std, num_layers_list, 'r'); legend_cell_array = [legend_cell_array, 'proposed method (std)'];
% ciplot(backprop_method.duration.mean - backprop_method.duration.std, backprop_method.duration.mean + backprop_method.duration.std, num_layers_list, 'b'); legend_cell_array = [legend_cell_array, 'backprop method (std)'];
% % ciplot(random_p_method.duration.mean - random_p_method.duration.std, random_p_method.duration.mean + random_p_method.duration.std, num_layers_list, 'b'); legend_cell_array = [legend_cell_array, 'random projection method (std)'];
% plot(num_layers_list, proposed_method.duration.mean, '-r^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'duration proposed method (mean)'];
% plot(num_layers_list, backprop_method.duration.mean, '-b^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'duration backprop method (mean)'];
% % plot(num_layers_list, random_p_method.duration.mean, '-b^', 'LineWidth', 2); legend_cell_array = [legend_cell_array, 'duration random projection method (mean)'];

% xticks(1 : numel(proposed_method.accuracy.mean));
% xlabel('# of Random Layers');
% ylabel('Time (sec)');
% hold off;
% title('Duration Comparison');
% legend(legend_cell_array, 'Location','northwest');

% suptitle(sprintf('%s - %d per layer', dataset, projected_dim));
