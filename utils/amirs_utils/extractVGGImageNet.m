% -------------------------------------------------------------------------
function extractVGGImageNet()
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


fprintf('[INFO] Loading data from pre-trained VGG ImageNet network...\n');
if (~exist('layers') || ~exist('meta'))
    load('/Users/amirhk/dev/data/imagenet-matconvnet-vgg-verydeep-16.mat');
end
fprintf('[INFO] Loading data successful!\n\n');
% -------------------------------------------------------------------------
% fprintf('[INFO] Extracting weights...\n');
% count = 0;
% for i = 1:numel(layers)
%     if (~isempty(layers{i}.weights))
%     % if (isfield(layers{i}, 'weights'))
%         count = count + 1;
%     end
% end
% fprintf('[INFO] Extracting weights successful!\n\n');
% -------------------------------------------------------------------------
% fprintf('[INFO] Number of weighted layers: %d\n', count);


fprintf('-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- \n\n');
fprintf('Input image size: ');
disp(meta.inputSize);
for i = 1:numel(layers)
    % if (~isempty(layers{i}.weights))
    if (strcmp(layers{i}.type, 'conv'))
        fprintf('-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- \n\n');
        fprintf('Layer %d', i);
        disp(layers{i}.weights);
        % W1 = layers{i}.weights{1};
        % W2 = layers{i}.weights{2};
        % save(sprintf('W1-layer-%d.mat', i), 'W1');
        % save(sprintf('W2-layer-%d.mat', i), 'W2');
    elseif (strcmp(layers{i}.type, 'relu'))
        fprintf('\trelu\n');
    elseif (strcmp(layers{i}.type, 'pool'))
        fprintf('\t');
        fprintf('pool - method: %s, stride: %d, pool: ', layers{i}.method, layers{i}.stride);
        disp(layers{i}.pool);
    end
end


% fprintf('-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- \n\n');
% fprintf('Input image size: ');
% disp(meta.inputSize);
% for i = 1:numel(layers)
%     fprintf('-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- \n\n');
%     disp(layers{i});
%     % % if (~isempty(layers{i}.weights))
%     % if (strcmp(layers{i}.type, 'conv'))
%     %     fprintf('-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- \n\n');
%     %     fprintf('Layer %d', i);
%     %     disp(layers{i}.weights);
%     % elseif (strcmp(layers{i}.type, 'relu'))
%     %     fprintf('\trelu\n');
%     % elseif (strcmp(layers{i}.type, 'pool'))
%     %     fprintf('\t');
%     %     fprintf('pool - method: %s, stride: %d, pool: ', layers{i}.method, layers{i}.stride);
%     %     disp(layers{i}.pool);
%     % end
% end
