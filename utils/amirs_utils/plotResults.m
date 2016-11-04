% dataDir = '/Volumes/Amir/results/';
% % subDataDir = 'oct 17-18 testing FC+0-5 w: input whitening = true, 1D dist sampling, kernel normalization = false';
% % subDataDir = 'oct 19-20 testing FC+0-5, input whitening = false, 1D dist sampling, kernel normalization = false';
% % subDataDir = 'oct 25-26 testing FC+0-5 with input whitening = true, 1D dist sampling, kernel normalization = false';
% % subDataDir = fullfile('oct 27-28 testing FC+0-5 with input whitening = true, comp rand weights, kernel normalization = false', 'batch 2');
% % subDataDir = 'oct 28 testing FC+0-5 without bottlenecks, comp rand weights, with weight decay';
% subDataDir = 'oct 29-30; FC+0-5; input whitening = T; 1D dist sampling; testing new lr';
% epochNum = 75;
% epochFile = sprintf('net-epoch-%d.mat', epochNum);
% fprintf('Loading files...'); i = 1;

% fc_plus_5 = load(fullfile(dataDir, subDataDir, 'cifar-alex-net-29-Oct-2016-21-07-25-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
% fc_plus_4 = load(fullfile(dataDir, subDataDir, 'cifar-alex-net-30-Oct-2016-00-39-57-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
% fc_plus_3 = load(fullfile(dataDir, subDataDir, 'cifar-alex-net-30-Oct-2016-03-56-50-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
% fc_plus_2 = load(fullfile(dataDir, subDataDir, 'cifar-alex-net-30-Oct-2016-06-17-49-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
% fc_plus_1 = load(fullfile(dataDir, subDataDir, 'cifar-alex-net-30-Oct-2016-08-18-17-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
% fc_only = load(fullfile(dataDir, subDataDir, 'cifar-alex-net-30-Oct-2016-10-05-26-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
% fprintf('\nDone!');

% figure;
% plot( ...
%   1:1:epochNum, [fc_only.info.train.error(1,:)], ... % [fc_only.stats.train.top1err], ... .info.train.error
%   1:1:epochNum, [fc_plus_1.info.train.error(1,:)], ... % [fc_plus_1.stats.train.top1err], ...
%   1:1:epochNum, [fc_plus_2.info.train.error(1,:)], ... % [fc_plus_2.stats.train.top1err], ...
%   1:1:epochNum, [fc_plus_3.info.train.error(1,:)], ... % [fc_plus_3.stats.train.top1err], ...
%   1:1:epochNum, [fc_plus_4.info.train.error(1,:)], ... % [fc_plus_4.stats.train.top1err], ...
%   1:1:epochNum, [fc_plus_5.info.train.error(1,:)], ... % [fc_plus_5.stats.train.top1err], ...
%   'LineWidth', 2);
% grid on
% title('Effects of Varying BackProp Depth on Training Accuracy');
% legend(...
%   'training on FC ONLY', ...
%   'training on FC + 1 blocks', ...
%   'training on FC + 2 blocks', ...
%   'training on FC + 3 blocks', ...
%   'training on FC + 4 blocks', ...
%   'training on FC + 5 blocks (FULL)');
% xlabel('epoch')
% ylabel('Training Error');

% figure;
% plot( ...
%   1:1:epochNum, [fc_only.info.val.error(1,:)], ... % [fc_only.stats.val.top1err], ...
%   1:1:epochNum, [fc_plus_1.info.val.error(1,:)], ... % [fc_plus_1.stats.val.top1err], ...
%   1:1:epochNum, [fc_plus_2.info.val.error(1,:)], ... % [fc_plus_2.stats.val.top1err], ...
%   1:1:epochNum, [fc_plus_3.info.val.error(1,:)], ... % [fc_plus_3.stats.val.top1err], ...
%   1:1:epochNum, [fc_plus_4.info.val.error(1,:)], ... % [fc_plus_4.stats.val.top1err], ...
%   1:1:epochNum, [fc_plus_5.info.val.error(1,:)], ... % [fc_plus_5.stats.val.top1err], ...
%   'LineWidth', 2);
% grid on
% title('Effects of Varying BackProp Depth on Validation Accuracy');
% legend(...
%   'training on FC ONLY', ...
%   'training on FC + 1 blocks', ...
%   'training on FC + 2 blocks', ...
%   'training on FC + 3 blocks', ...
%   'training on FC + 4 blocks', ...
%   'training on FC + 5 blocks (FULL)');
% xlabel('epoch')
% ylabel('Validation Error');


% == == == == == == == == == == == == == == == == == == == == == == == == == ==
% == == == == == == == == == == == == == == == == == == == == == == == == == ==


% dataDir = '/Volumes/Amir-1/results/';
% subDataDir = '2016-11-1-1; FC+5; compRand weights; CIFAR portion = 50; testing different weight decays';
% epochNum = 50;
% epochFile = sprintf('net-epoch-%d.mat', epochNum);
% fprintf('Loading files...'); i = 1;


% weight_decay_1e_1 = load(fullfile(dataDir, subDataDir, 'cifar-alex-net-1-Nov-2016-15-37-00-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
% weight_decay_1e_2 = load(fullfile(dataDir, subDataDir, 'cifar-alex-net-1-Nov-2016-16-46-28-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
% weight_decay_1e_3 = load(fullfile(dataDir, subDataDir, 'cifar-alex-net-1-Nov-2016-17-57-49-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
% weight_decay_1e_4 = load(fullfile(dataDir, subDataDir, 'cifar-alex-net-1-Nov-2016-19-08-39-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
% weight_decay_0 = load(fullfile(dataDir, subDataDir, 'cifar-alex-net-1-Nov-2016-20-18-55-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
% fprintf('\nDone!');

% figure;
% plot( ...
%   1:1:epochNum, [weight_decay_0.info.train.error(1,:)], ...
%   1:1:epochNum, [weight_decay_1e_4.info.train.error(1,:)], ...
%   1:1:epochNum, [weight_decay_1e_3.info.train.error(1,:)], ...
%   1:1:epochNum, [weight_decay_1e_2.info.train.error(1,:)], ...
%   1:1:epochNum, [weight_decay_1e_1.info.train.error(1,:)], ...
%   'LineWidth', 2);
% grid on
% title('Effects of Varying Weight Decay on Training Accuracy');
% legend(...
%   'weight decay = 0', ...
%   'weight decay = 1e-4', ...
%   'weight decay = 1e-3', ...
%   'weight decay = 1e-2', ...
%   'weight decay = 1e-1');
% xlabel('epoch')
% ylabel('Training Error');

% figure;
% plot( ...
%   1:1:epochNum, [weight_decay_0.info.val.error(1,:)], ...
%   1:1:epochNum, [weight_decay_1e_4.info.val.error(1,:)], ...
%   1:1:epochNum, [weight_decay_1e_3.info.val.error(1,:)], ...
%   1:1:epochNum, [weight_decay_1e_2.info.val.error(1,:)], ...
%   1:1:epochNum, [weight_decay_1e_1.info.val.error(1,:)], ...
%   'LineWidth', 2);
% grid on
% title('Effects of Varying Weight Decay on Validation Accuracy');
% legend(...
%   'weight decay = 0', ...
%   'weight decay = 1e-4', ...
%   'weight decay = 1e-3', ...
%   'weight decay = 1e-2', ...
%   'weight decay = 1e-1');
% xlabel('epoch')
% ylabel('Validation Error');


% == == == == == == == == == == == == == == == == == == == == == == == == == ==
% == == == == == == == == == == == == == == == == == == == == == == == == == ==


dataDir = '/Volumes/Amir-1/results/';
subDataDir = '2016-11-01-02; AlexNet; testing different bottle neck compresson ratios; layers 1, 2 ';
epochNum = 50;
epochFile = sprintf('net-epoch-%d.mat', epochNum);
fprintf('Loading files...'); i = 1;


bottle_neck_divide_by_1 = load(fullfile(dataDir, subDataDir, 'cifar-alex-net-bottle-neck-1-Nov-2016-23-50-31-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
bottle_neck_divide_by_2 = load(fullfile(dataDir, subDataDir, 'cifar-alex-net-bottle-neck-2-Nov-2016-03-25-34-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
bottle_neck_divide_by_4 = load(fullfile(dataDir, subDataDir, 'cifar-alex-net-bottle-neck-2-Nov-2016-06-02-52-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
bottle_neck_divide_by_8 = load(fullfile(dataDir, subDataDir, 'cifar-alex-net-bottle-neck-2-Nov-2016-08-14-13-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;
bottle_neck_divide_by_16 = load(fullfile(dataDir, subDataDir, 'cifar-alex-net-bottle-neck-2-Nov-2016-10-17-47-GPU2', epochFile)); fprintf('\t%d', i); i = i + 1;

fprintf('\nDone!');

figure;
plot( ...
  1:1:epochNum, [bottle_neck_divide_by_1.info.train.error(1,:)], ...
  1:1:epochNum, [bottle_neck_divide_by_2.info.train.error(1,:)], ...
  1:1:epochNum, [bottle_neck_divide_by_4.info.train.error(1,:)], ...
  1:1:epochNum, [bottle_neck_divide_by_8.info.train.error(1,:)], ...
  1:1:epochNum, [bottle_neck_divide_by_16.info.train.error(1,:)], ...
  'LineWidth', 2);
grid on
title('Effects of Varying Bottle Neck Compresson on Training Accuracy');
legend(...
  'bottle neck divide by = 1', ...
  'bottle neck divide by = 2', ...
  'bottle neck divide by = 4', ...
  'bottle neck divide by = 8', ...
  'bottle neck divide by = 16');
xlabel('epoch')
ylabel('Training Error');

figure;
plot( ...
  1:1:epochNum, [bottle_neck_divide_by_1.info.val.error(1,:)], ...
  1:1:epochNum, [bottle_neck_divide_by_2.info.val.error(1,:)], ...
  1:1:epochNum, [bottle_neck_divide_by_4.info.val.error(1,:)], ...
  1:1:epochNum, [bottle_neck_divide_by_8.info.val.error(1,:)], ...
  1:1:epochNum, [bottle_neck_divide_by_16.info.val.error(1,:)], ...
  'LineWidth', 2);
grid on
title('Effects of Varying Bottle Neck Compresson on Validation Accuracy');
legend(...
  'bottle neck divide by = 1', ...
  'bottle neck divide by = 2', ...
  'bottle neck divide by = 4', ...
  'bottle neck divide by = 8', ...
  'bottle neck divide by = 16');
xlabel('epoch')
ylabel('Validation Error');


% % == == == == == == == == == == == == == == == == == == == == == == == == == ==
% % == == == == == == == == == == == == == == == == == == == == == == == == == ==


% dataDir = '/Volumes/Amir-1/results/';
% subDataDir = '2016-11-03-03; LeNet; testing different bottle neck compresson ratios; layers 1 & layers 1, 2 ';
% epochNum = 20;
% epochFile = sprintf('net-epoch-%d.mat', epochNum);
% fprintf('Loading files...'); i = 1;

% cifar_lenet_1_1 = load(fullfile(dataDir, subDataDir, 'cifar-lenet-1-1', epochFile)); fprintf('\t%d', i); i = i + 1;
% cifar_lenet_1_2 = load(fullfile(dataDir, subDataDir, 'cifar-lenet-1-2', epochFile)); fprintf('\t%d', i); i = i + 1;
% cifar_lenet_1_4 = load(fullfile(dataDir, subDataDir, 'cifar-lenet-1-4', epochFile)); fprintf('\t%d', i); i = i + 1;
% cifar_lenet_1_8 = load(fullfile(dataDir, subDataDir, 'cifar-lenet-1-8', epochFile)); fprintf('\t%d', i); i = i + 1;
% cifar_lenet_1_16 = load(fullfile(dataDir, subDataDir, 'cifar-lenet-1-16', epochFile)); fprintf('\t%d', i); i = i + 1;
% cifar_lenet_2_1 = load(fullfile(dataDir, subDataDir, 'cifar-lenet-2-1', epochFile)); fprintf('\t%d', i); i = i + 1;
% cifar_lenet_2_2 = load(fullfile(dataDir, subDataDir, 'cifar-lenet-2-2', epochFile)); fprintf('\t%d', i); i = i + 1;
% cifar_lenet_2_4 = load(fullfile(dataDir, subDataDir, 'cifar-lenet-2-4', epochFile)); fprintf('\t%d', i); i = i + 1;
% cifar_lenet_2_8 = load(fullfile(dataDir, subDataDir, 'cifar-lenet-2-8', epochFile)); fprintf('\t%d', i); i = i + 1;
% cifar_lenet_2_16 = load(fullfile(dataDir, subDataDir, 'cifar-lenet-2-16', epochFile)); fprintf('\t%d', i); i = i + 1;
% cifar_lenet_no_bottle_neck = load(fullfile(dataDir, subDataDir, 'cifar-lenet-no-bottle-neck', epochFile)); fprintf('\t%d', i); i = i + 1;

% fprintf('\nDone!');

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % number of bottle necks = 1
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% getCompressionRate = @(k) (96 + 64*(32+k))/(3168);

% h = figure;
% plot( ...
%   1:1:epochNum, [cifar_lenet_1_1.info.train.error(1,:)], ...
%   1:1:epochNum, [cifar_lenet_1_2.info.train.error(1,:)], ...
%   1:1:epochNum, [cifar_lenet_1_4.info.train.error(1,:)], ...
%   1:1:epochNum, [cifar_lenet_1_8.info.train.error(1,:)], ...
%   1:1:epochNum, [cifar_lenet_1_16.info.train.error(1,:)], ...
%   1:1:epochNum, [cifar_lenet_no_bottle_neck.info.train.error(1,:)], ...
%   'k--', ...
%   'LineWidth', 2);
% grid on
% title('Effects of Varying Bottle Neck Compression on Training Accuracy');
% legend(...
%   sprintf('bottle-neck count = 1; k = 1, [conv] size: %3.2f', getCompressionRate(1)), ...
%   sprintf('bottle-neck count = 1; k = 2, [conv] size: %3.2f', getCompressionRate(2)), ...
%   sprintf('bottle-neck count = 1; k = 4, [conv] size: %3.2f', getCompressionRate(4)), ...
%   sprintf('bottle-neck count = 1; k = 8, [conv] size: %3.2f', getCompressionRate(8)), ...
%   sprintf('bottle-neck count = 1; k = 16, [conv] size: %3.2f', getCompressionRate(16)), ...
%   sprintf('NO bottle-neck[conv] size: %3.2f', 1.00));
% xlabel('epoch')
% ylabel('Training Error');
% saveas(h,'Training Comparison - 1 bottle necks.png')

% h = figure;
% plot( ...
%   1:1:epochNum, [cifar_lenet_1_1.info.val.error(1,:)], ...
%   1:1:epochNum, [cifar_lenet_1_2.info.val.error(1,:)], ...
%   1:1:epochNum, [cifar_lenet_1_4.info.val.error(1,:)], ...
%   1:1:epochNum, [cifar_lenet_1_8.info.val.error(1,:)], ...
%   1:1:epochNum, [cifar_lenet_1_16.info.val.error(1,:)], ...
%   1:1:epochNum, [cifar_lenet_no_bottle_neck.info.val.error(1,:)], ...
%   'k--', ...
%   'LineWidth', 2);
% grid on
% title('Effects of Varying Bottle Neck Compression on Validation Accuracy');
% legend(...
%   sprintf('bottle-neck count = 1; k = 1, [conv] size: %3.2f', getCompressionRate(1)), ...
%   sprintf('bottle-neck count = 1; k = 2, [conv] size: %3.2f', getCompressionRate(2)), ...
%   sprintf('bottle-neck count = 1; k = 4, [conv] size: %3.2f', getCompressionRate(4)), ...
%   sprintf('bottle-neck count = 1; k = 8, [conv] size: %3.2f', getCompressionRate(8)), ...
%   sprintf('bottle-neck count = 1; k = 16, [conv] size: %3.2f', getCompressionRate(16)), ...
%   sprintf('NO bottle-neck[conv] size: %3.2f', 1.00));
% xlabel('epoch')
% ylabel('Validation Error');
% saveas(h,'Validation Comparison - 1 bottle necks.png')

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % number of bottle necks = 2
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% getCompressionRate = @(k) (96 + 160*k)/(3168);
% h = figure;
% plot( ...
%   1:1:epochNum, [cifar_lenet_2_1.info.train.error(1,:)], ...
%   1:1:epochNum, [cifar_lenet_2_2.info.train.error(1,:)], ...
%   1:1:epochNum, [cifar_lenet_2_4.info.train.error(1,:)], ...
%   1:1:epochNum, [cifar_lenet_2_8.info.train.error(1,:)], ...
%   1:1:epochNum, [cifar_lenet_2_16.info.train.error(1,:)], ...
%   1:1:epochNum, [cifar_lenet_no_bottle_neck.info.train.error(1,:)], ...
%   'k--', ...
%   'LineWidth', 2);
% grid on
% title('Effects of Varying Bottle Neck Compression on Training Accuracy');
% legend(...
%   sprintf('bottle-neck count = 2; k = 1, [conv] size: %3.2f', getCompressionRate(1)), ...
%   sprintf('bottle-neck count = 2; k = 2, [conv] size: %3.2f', getCompressionRate(2)), ...
%   sprintf('bottle-neck count = 2; k = 4, [conv] size: %3.2f', getCompressionRate(4)), ...
%   sprintf('bottle-neck count = 2; k = 8, [conv] size: %3.2f', getCompressionRate(8)), ...
%   sprintf('bottle-neck count = 2; k = 16, [conv] size: %3.2f', getCompressionRate(16)), ...
%   sprintf('NO bottle-neck[conv] size: %3.2f', 1.00));
% xlabel('epoch')
% ylabel('Training Error');
% saveas(h,'Training Comparison - 2 bottle necks.png')

% h = figure;
% plot( ...
%   1:1:epochNum, [cifar_lenet_2_1.info.val.error(1,:)], ...
%   1:1:epochNum, [cifar_lenet_2_2.info.val.error(1,:)], ...
%   1:1:epochNum, [cifar_lenet_2_4.info.val.error(1,:)], ...
%   1:1:epochNum, [cifar_lenet_2_8.info.val.error(1,:)], ...
%   1:1:epochNum, [cifar_lenet_2_16.info.val.error(1,:)], ...
%   1:1:epochNum, [cifar_lenet_no_bottle_neck.info.val.error(1,:)], ...
%   'k--', ...
%   'LineWidth', 2);
% grid on
% title('Effects of Varying Bottle Neck Compression on Validation Accuracy');
% legend(...
%   sprintf('bottle-neck count = 2; k = 1, [conv] size: %3.2f', getCompressionRate(1)), ...
%   sprintf('bottle-neck count = 2; k = 2, [conv] size: %3.2f', getCompressionRate(2)), ...
%   sprintf('bottle-neck count = 2; k = 4, [conv] size: %3.2f', getCompressionRate(4)), ...
%   sprintf('bottle-neck count = 2; k = 8, [conv] size: %3.2f', getCompressionRate(8)), ...
%   sprintf('bottle-neck count = 2; k = 16, [conv] size: %3.2f', getCompressionRate(16)), ...
%   sprintf('NO bottle-neck[conv] size: %3.2f', 1.00));
% xlabel('epoch')
% ylabel('Validation Error');
% saveas(h,'Validation Comparison - 2 bottle necks.png')
