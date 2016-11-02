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
subDataDir = '2016-11-01-02; testing different bottle neck compresson ratios; layers 1';
epochNum = 50;
epochFile = sprintf('net-epoch-%d.mat', epochNum);
fprintf('Loading files...'); i = 1;


bottle_neck_divide_by_1 = load(fullfile(dataDir, subDataDir, 'cifar-alex-net-bottle-neck-1-Nov-2016-23-48-47-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
bottle_neck_divide_by_2 = load(fullfile(dataDir, subDataDir, 'cifar-alex-net-bottle-neck-2-Nov-2016-02-45-47-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
bottle_neck_divide_by_4 = load(fullfile(dataDir, subDataDir, 'cifar-alex-net-bottle-neck-2-Nov-2016-05-03-09-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
bottle_neck_divide_by_8 = load(fullfile(dataDir, subDataDir, 'cifar-alex-net-bottle-neck-2-Nov-2016-07-05-01-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
bottle_neck_divide_by_16 = load(fullfile(dataDir, subDataDir, 'cifar-alex-net-bottle-neck-2-Nov-2016-09-01-44-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;
bottle_neck_divide_by_32 = load(fullfile(dataDir, subDataDir, 'cifar-alex-net-bottle-neck-2-Nov-2016-10-54-09-GPU1', epochFile)); fprintf('\t%d', i); i = i + 1;

fprintf('\nDone!');

figure;
plot( ...
  1:1:epochNum, [bottle_neck_divide_by_1.info.train.error(1,:)], ...
  1:1:epochNum, [bottle_neck_divide_by_2.info.train.error(1,:)], ...
  1:1:epochNum, [bottle_neck_divide_by_4.info.train.error(1,:)], ...
  1:1:epochNum, [bottle_neck_divide_by_8.info.train.error(1,:)], ...
  1:1:epochNum, [bottle_neck_divide_by_16.info.train.error(1,:)], ...
  1:1:epochNum, [bottle_neck_divide_by_32.info.train.error(1,:)], ...
  'LineWidth', 2);
grid on
title('Effects of Varying Weight Decay on Training Accuracy');
legend(...
  'bottle neck divide by = 1', ...
  'bottle neck divide by = 2', ...
  'bottle neck divide by = 4', ...
  'bottle neck divide by = 8', ...
  'bottle neck divide by = 16', ...
  'bottle neck divide by = 32');
xlabel('epoch')
ylabel('Training Error');

figure;
plot( ...
  1:1:epochNum, [bottle_neck_divide_by_1.info.val.error(1,:)], ...
  1:1:epochNum, [bottle_neck_divide_by_2.info.val.error(1,:)], ...
  1:1:epochNum, [bottle_neck_divide_by_4.info.val.error(1,:)], ...
  1:1:epochNum, [bottle_neck_divide_by_8.info.val.error(1,:)], ...
  1:1:epochNum, [bottle_neck_divide_by_16.info.val.error(1,:)], ...
  1:1:epochNum, [bottle_neck_divide_by_32.info.val.error(1,:)], ...
  'LineWidth', 2);
grid on
title('Effects of Varying Weight Decay on Validation Accuracy');
legend(...
  'bottle neck divide by = 1', ...
  'bottle neck divide by = 2', ...
  'bottle neck divide by = 4', ...
  'bottle neck divide by = 8', ...
  'bottle neck divide by = 16', ...
  'bottle neck divide by = 32');
xlabel('epoch')
ylabel('Validation Error');
