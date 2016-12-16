function main_cnn_rusboost()


  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 0. some important parameter definition
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  T = 3; % number of boosting iterations
  % E = 50; % number of epochs() % TODO: currently can't be used...
  backpropDepth = 4;
  weightInitSource = 'gen';  % {'load' | 'gen'}
  weightInitSequence = {'compRand', 'compRand', 'compRand'};
  random_undersampling_ratio = (65/35);


  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 1. take as input a pre-processed IMDB (augment cancer in training set, that's it!), say
  %   train: 94 patients
  %   test: 10 patients, ~1000 health, ~20 cancer
  % TODO: this can be extended to be say 10-fold ensemble larp, then average the folds
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  fprintf('[INFO]: Loading saved imdb... ');
  imdb = load(fullfile(getDevPath(), '/matconvnet/data_1/_prostate/_saved_prostate_imdb.mat'));
  imdb = imdb.imdb;
  fprintf('done!\n');
  % fprintf('[INFO]: TRAINING SET data distribution:\n');
  % tabulate(imdb.images.labels(imdb.images.set == 1))
  % fprintf('[INFO]: TESTING SET data distribution:\n');
  % tabulate(imdb.images.labels(imdb.images.set == 3))

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 2. process the imdb to separate positive and negative samples (to be
  % randomly-undersampled later)
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  data_train = imdb.images.data(:,:,:,imdb.images.set == 1);
  labels_train = imdb.images.labels(imdb.images.set == 1);
  data_train_healthy = data_train(:,:,:,labels_train == 1);
  data_train_cancer = data_train(:,:,:,labels_train == 2);
  data_train_count = size(data_train, 4);
  data_train_healthy_count = size(data_train_healthy, 4);
  data_train_cancer_count = size(data_train_cancer, 4);

  data_test = imdb.images.data(:,:,:,imdb.images.set == 3);
  labels_test = imdb.images.labels(imdb.images.set == 3);
  data_test_healthy = data_test(:,:,:,labels_test == 1);
  data_test_cancer = data_test(:,:,:,labels_test == 2);
  data_test_count = size(data_test, 4);
  data_test_healthy_count = size(data_test_healthy, 4);
  data_test_cancer_count = size(data_test_cancer, 4);

  fprintf('[INFO]: TRAINING SET:\n');
  fprintf('\t total: %d\n', data_train_count);
  fprintf('\t healthy: %d\n', data_train_healthy_count);
  fprintf('\t cancer: %d\n', data_train_cancer_count);
  fprintf('[INFO]: TESTING SET:\n');
  fprintf('\t total: %d\n', data_test_count);
  fprintf('\t healthy: %d\n', data_test_healthy_count);
  fprintf('\t cancer: %d\n', data_test_cancer_count);

  % m = size(TRAIN, 1);
  % POS_DATA = TRAIN(TRAIN(:, end) == 1, :);
  % NEG_DATA = TRAIN(TRAIN(:, end) == 0, :);
  % pos_size = size(POS_DATA, 1);
  % neg_size = size(NEG_DATA, 1);

  % Reorganize TRAIN by putting all the positive and negative examples
  % together, respectively.
  % TODO: WHY???!!!?!?!?
  % TRAIN = [POS_DATA; NEG_DATA];


  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 3. initialize training sample weights
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % W stores the weights of the instances in each row for every iteration of
  % boosting. Weights for all the instances are initialized by 1/m for the
  % first iteration.
  % TODO: Make sure when you shuffle shit... these weights are also shuffled!!!
  % same with undersampling!!
  W = zeros(1, data_train_count);
  for i = 1 : data_train_count
      W(1, i) = 1 / data_train_count;
  end

  % L stores pseudo loss values, H stores hypothesis, B stores (1/beta)
  % values that is used as the weight of the % hypothesis while forming the
  % final hypothesis. % All of the following are of length <=T and stores
  % values for every iteration of the boosting process.
  L = [];
  H = {};
  B = [];

  % Loop counter
  t = 1;

  % Keeps counts of the number of times the same boosting iteration have been
  % repeated
  count = 0;

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 4. go through T iterations of RUSBoost, each of which trains a CNN over E epochs
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 5. take output of CNN after E epochs, and adjust weights of each sample accordingly
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 6. repeat steps 4 & 5 T times, saving model (to run test data on later), and beta for each
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  training_test_imdb.images.data = data_train;
  training_test_imdb.images.labels = labels_train;
  training_test_imdb.images.set = 3 * ones(length(labels_train), 1);
  training_test_imdb.meta.sets = {'train', 'val', 'test'};

  fprintf('\n-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- \n');

  while t <= T

      fprintf('\n[INFO] Boosting iteration #%d...\n', t);

      % Resampling NEG_DATA with weights of positive example
      % TODO: oversample / repeat training samples based on W(t - 1)
      fprintf('\t[INFO] Resampling healthy and cancer data (ratio = 65/35)... ');
      initial_data_train_healthy_count = data_train_healthy_count;
      resampled_data_train_healthy_count = round(data_train_cancer_count * random_undersampling_ratio);
      resampled_data_train_healthy_indices = randsample(1:initial_data_train_healthy_count, resampled_data_train_healthy_count, false);
      resampled_data_train_healthy = data_train_healthy(:,:,:, resampled_data_train_healthy_indices);
      resampled_data_train_all = cat(4, data_train_cancer, resampled_data_train_healthy);
      resampled_labels_train_all = cat(2, 2 * ones(1, data_train_cancer_count), 1 * ones(1, size(resampled_data_train_healthy, 4)));
      % tabulate(resampled_labels_train_all)
      fprintf('done!\n');

      % Shuffle this to mixup order of healthy and cancer in imdb so we don't
      % have the CNN overtrain in 1 particular direction. Only shuffling for
      % training; later weights are calculated and updated for all training data.
      ix = randperm(size(resampled_data_train_all, 4));
      new_data = resampled_data_train_all(:,:,:,ix);
      new_labels = resampled_labels_train_all(ix);

      training_resampled_imdb.images.data = new_data;
      training_resampled_imdb.images.labels = single(new_labels);
      training_resampled_imdb.images.set = 1 * ones(length(new_labels), 1);
      training_resampled_imdb.meta.sets = {'train', 'val', 'test'};

      % Weird. Need at least 1 test sample for cnn_train to work. Ignore
      training_resampled_imdb.images.data = cat(4, training_resampled_imdb.images.data, new_data(:,:,:, end));
      training_resampled_imdb.images.labels = cat(2,training_resampled_imdb.images.labels, new_labels(end));
      training_resampled_imdb.images.set = cat(1, training_resampled_imdb.images.set, 3);

      fprintf('\t[INFO] Training model (healthy: %d, cancer: %d)...\n', size(resampled_data_train_healthy, 4), data_train_cancer_count);
      [net, info] = cnn_amir( ...
        'imdb', training_resampled_imdb, ...
        'dataset', 'prostate', ...
        'networkArch', 'prostatenet', ...
        'backpropDepth', backpropDepth, ...
        'weightInitSource', weightInitSource, ...
        'weightInitSequence', weightInitSequence, ...
        'debugFlag', false);
      % fprintf('\tdone!\n');

      fprintf('\t[INFO] Getting predictions (healthy: %d, cancer: %d)...\n', data_train_healthy_count, data_train_cancer_count);
      % IMPORTANT NOTE: we randomly undersample when training a model, but then,
      % we use all of the training samples (in their order) to update weights.
      predictions = getPredictionsFromNetOnImdb(net, training_test_imdb);
      % fprintf('done!\n');

      % Computing the pseudo loss of hypothesis 'model'
      fprintf('\t[INFO] Computing pseudo loss... ');
      loss = 0;
      for i = 1:data_train_count
          if labels_train(i) == predictions(i)
              continue;
          else
              loss = loss + W(t, i);
          end
      end
      fprintf('Loss: %6.5f\n', loss);

      % If count exceeds a pre-defined threshold (5 in the current
      % implementation), the loop is broken and rolled back to the state
      % where loss > 0.5 was not encountered.
      if count > 5
         L = L(1:t-1);
         H = H(1:t-1);
         B = B(1:t-1);
         fprintf('\tToo many iterations have loss > 0.5\n');
         fprintf('\tAborting boosting...\n');
         break;
      end

      % If the loss is greater than 1/2, it means that an inverted
      % hypothesis would perform better. In such cases, do not take that
      % hypothesis into consideration and repeat the same iteration. 'count'
      % keeps counts of the number of times the same boosting iteration have
      % been repeated
      if loss > 0.5
          count = count + 1;
          continue;
      else
          count = 1;
      end

      H{t} = net; % Hypothesis function / Trained CNN Network
      L(t) = loss; % Pseudo-loss at each iteration
      beta = loss / (1 - loss); % Setting weight update parameter 'beta'.
      B(t) = log(1 / beta); % Weight of the hypothesis

      % At the final iteration there is no need to update the weights any
      % further
      if t == T
          break;
      end

      % Updating weight
      fprintf('\t[INFO] Updating weights... ');
      for i = 1:data_train_count
          if labels_train(i) == predictions(i)
              W(t + 1, i) = W(t, i) * beta;
          else
              W(t + 1, i) = W(t, i);
          end
      end
      fprintf('done!\n');

      % Normalizing the weight for the next iteration
      sum_W = sum(W(t + 1, :));
      for i = 1:data_train_count
          W(t + 1, i) = W(t + 1, i) / sum_W;
      end

      % Incrementing loop counter
      t = t + 1;
  end

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 7. test on test set, keeping in mind beta's between each mode
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  % The final hypothesis is calculated and tested on the test set
  % simulteneously.

  % Normalizing B
  sum_B = sum(B);
  for i = 1:size(B,2)
     B(i) = B(i) / sum_B;
  end

  test_imdb.images.data = data_test;
  test_imdb.images.labels = labels_test;
  test_imdb.images.set = 3 * ones(length(labels_test), 1);
  test_imdb.meta.sets = {'train', 'val', 'test'};

  test_set_prediction_overall = zeros(data_test_count, 2);

  fprintf('\n-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- \n');

  test_set_predictions_per_model = {};
  for i = 1:size(H, 2) % looping through all trained networks
    fprintf('\n[INFO] Getting test set predictions for model #%d (healthy: %d, cancer: %d)...\n', i, data_test_healthy_count, data_test_cancer_count);
    net = H{i};
    test_set_predictions_per_model{i} = getPredictionsFromNetOnImdb(net, test_imdb);
  end

  for i = 1:data_test_count
    % Calculating the total weight of the class labels from all the models
    % produced during boosting
    wt_healthy = 0; % class 1
    wt_cancer = 0; % class 2
    for j = 1:size(H, 2) % looping through all trained networks
       p = test_set_predictions_per_model{j}(1);
       if p == 2 % if is cancer
           wt_cancer = wt_cancer + B(j);
       else
           wt_healthy = wt_healthy + B(j);
       end
    end

    if (wt_cancer > wt_healthy)
        test_set_prediction_overall(i,:) = [2 wt_cancer];
    else
        test_set_prediction_overall(i,:) = [1 wt_cancer]; % TODO: should this not be wt_healthy?!!?!?!
    end
  end
  predictions_test = test_set_prediction_overall(:, 1)';

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 8. done, go treat yourself to something sugary!
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  fprintf('\n-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- \n');
  fprintf('\njigar talaaaaaa!!!\n');
  TP = sum((labels_test == predictions_test) .* (predictions_test == 2)); % TP
  TN = sum((labels_test == predictions_test) .* (predictions_test == 1)); % TN
  FP = sum((labels_test ~= predictions_test) .* (predictions_test == 2)); % FP
  FN = sum((labels_test ~= predictions_test) .* (predictions_test == 1)); % FN
  fprintf('TP: %d\n', TP);
  fprintf('TN: %d\n', TN);
  fprintf('FP: %d\n', FP);
  fprintf('FN: %d\n', FN);
  fprintf('\n\n-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- \n\n');
  fprintf('Acc: %3.2f\n', (TP + TN) / (TP + TN + FP + FN));
  fprintf('Sens: %3.2f\n', TP / (TP + FN));
  fprintf('Spec: %3.2f\n', TN / (TN + FP));
  fprintf('\n\n-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- \n');










% -------------------------------------------------------------------------
function fn = getBatch()
% -------------------------------------------------------------------------
  fn = @(x,y) getSimpleNNBatch(x,y);

% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
  images = imdb.images.data(:,:,:,batch);
  labels = imdb.images.labels(1,batch);
  if rand > 0.5, images=fliplr(images); end

% -------------------------------------------------------------------------
function predictions = getPredictionsFromNetOnImdb(net, imdb)
% -------------------------------------------------------------------------
  [net, info] = cnn_train(net, imdb, getBatch(), ...
    'errorFunction', 'multiclass-prostate', ...
    'debugFlag', false, ...
    'continue', false, ...
    'numEpochs', 1, ...
    'val', find(imdb.images.set == 3));
  predictions = info.predictions;




% % This function implements the RUSBoost Algorithm. For more details on the
% % theoretical description of the algorithm please refer to the following
% % paper:
% % C. Seiffert, T.M. Khoshgoftaar, J. Van Hulse and A. Napolitano,
% % "RUSBoost: A Hybrid Approach to Alleviating Class Imbalance, IEEE
% % Transaction on Systems, Man and Cybernetics-Part A: Systems and Human,
% % Vol.40(1), January 2010.
% % Input: TRAIN = Training data as matrix
% %        TEST = Test data as matrix
% %        WeakLearn = String to choose algortihm. Choices are
% %                    'svm','tree','knn' and 'logistic'.
% % Output: prediction = size(TEST,1)x 2 matrix. Col 1 is class labels for
% %                      all instances. Col 2 is probability of the instances
% %                      being classified as positive class.


% javaaddpath('weka.jar');

% %% Training RUSBoost
% % Total number of instances in the training set
% m = size(TRAIN,1);
% POS_DATA = TRAIN(TRAIN(:,end)==1,:);
% NEG_DATA = TRAIN(TRAIN(:,end)==0,:);
% pos_size = size(POS_DATA,1);
% neg_size = size(NEG_DATA,1);

% % Reorganize TRAIN by putting all the positive and negative exampels
% % together, respectively.
% TRAIN = [POS_DATA;NEG_DATA];

% % Converting training set into Weka compatible format
% CSVtoARFF (TRAIN, 'train', 'train');
% train_reader = javaObject('java.io.FileReader', 'train.arff');
% train = javaObject('weka.core.Instances', train_reader);
% train.setClassIndex(train.numAttributes() - 1);

% % 65% of NEG_DATA
% neg65 = round(pos_size * (65/35));

% % Total number of iterations of the boosting method
% T = 10;

% % W stores the weights of the instances in each row for every iteration of
% % boosting. Weights for all the instances are initialized by 1/m for the
% % first iteration.
% W = zeros(1,m);
% for i = 1:m
%     W(1,i) = 1/m;
% end

% % L stores pseudo loss values, H stores hypothesis, B stores (1/beta)
% % values that is used as the weight of the % hypothesis while forming the
% % final hypothesis. % All of the following are of length <=T and stores
% % values for every iteration of the boosting process.
% L = [];
% H = {};
% B = [];

% % Loop counter
% t = 1;

% % Keeps counts of the number of times the same boosting iteration have been
% % repeated
% count = 0;

% % Boosting T iterations
% while t <= T

%     % LOG MESSAGE
%     disp (['Boosting iteration #' int2str(t)]);

%     % Resampling NEG_DATA with weights of positive example
%     RESAM_NEG = NEG_DATA(randsample(1:neg_size,neg65,false),:);
%     RESAMPLED = [POS_DATA;RESAM_NEG];

%     % Converting resample training set into Weka compatible format
%     CSVtoARFF (RESAMPLED,'resampled','resampled');
%     reader = javaObject('java.io.FileReader','resampled.arff');
%     resampled = javaObject('weka.core.Instances',reader);
%     resampled.setClassIndex(resampled.numAttributes()-1);

%     % Training a weak learner. 'pred' is the weak hypothesis. However, the
%     % hypothesis function is encoded in 'model'.
%     switch WeakLearn
%         case 'svm'
%             model = javaObject('weka.classifiers.functions.SMO');
%         case 'tree'
%             model = javaObject('weka.classifiers.trees.J48');
%         case 'knn'
%             model = javaObject('weka.classifiers.lazy.IBk');
%             model.setKNN(5);
%         case 'logistic'
%             model = javaObject('weka.classifiers.functions.Logistic');
%     end
%     model.buildClassifier(resampled);

%     pred = zeros(m,1);
%     for i = 0 : m - 1
%         pred(i+1) = model.classifyInstance(train.instance(i));
%     end

%     % Computing the pseudo loss of hypothesis 'model'
%     loss = 0;
%     for i = 1:m
%         if TRAIN(i,end)==pred(i)
%             continue;
%         else
%             loss = loss + W(t,i);
%         end
%     end

%     % If count exceeds a pre-defined threshold (5 in the current
%     % implementation), the loop is broken and rolled back to the state
%     % where loss > 0.5 was not encountered.
%     if count > 5
%        L = L(1:t-1);
%        H = H(1:t-1);
%        B = B(1:t-1);
%        disp ('Too many iterations have loss > 0.5');
%        disp ('Aborting boosting...');
%        break;
%     end

%     % If the loss is greater than 1/2, it means that an inverted
%     % hypothesis would perform better. In such cases, do not take that
%     % hypothesis into consideration and repeat the same iteration. 'count'
%     % keeps counts of the number of times the same boosting iteration have
%     % been repeated
%     if loss > 0.5
%         count = count + 1;
%         continue;
%     else
%         count = 1;
%     end

%     L(t) = loss; % Pseudo-loss at each iteration
%     H{t} = model; % Hypothesis function
%     beta = loss/(1-loss); % Setting weight update parameter 'beta'.
%     B(t) = log(1/beta); % Weight of the hypothesis

%     % At the final iteration there is no need to update the weights any
%     % further
%     if t==T
%         break;
%     end

%     % Updating weight
%     for i = 1:m
%         if TRAIN(i,end)==pred(i)
%             W(t+1,i) = W(t,i)*beta;
%         else
%             W(t+1,i) = W(t,i);
%         end
%     end

%     % Normalizing the weight for the next iteration
%     sum_W = sum(W(t+1,:));
%     for i = 1:m
%         W(t+1,i) = W(t+1,i)/sum_W;
%     end

%     % Incrementing loop counter
%     t = t + 1;
% end

% % The final hypothesis is calculated and tested on the test set
% % simulteneously.

% %% Testing RUSBoost
% n = size(TEST,1); % Total number of instances in the test set

% CSVtoARFF(TEST,'test','test');
% test = 'test.arff';
% test_reader = javaObject('java.io.FileReader', test);
% test = javaObject('weka.core.Instances', test_reader);
% test.setClassIndex(test.numAttributes() - 1);

% % Normalizing B
% sum_B = sum(B);
% for i = 1:size(B,2)
%    B(i) = B(i)/sum_B;
% end

% prediction = zeros(n,2);

% for i = 1:n
%     % Calculating the total weight of the class labels from all the models
%     % produced during boosting
%     wt_zero = 0;
%     wt_one = 0;
%     for j = 1:size(H,2)
%        p = H{j}.classifyInstance(test.instance(i-1));
%        if p==1
%            wt_one = wt_one + B(j);
%        else
%            wt_zero = wt_zero + B(j);
%        end
%     end

%     if (wt_one > wt_zero)
%         prediction(i,:) = [1 wt_one];
%     else
%         prediction(i,:) = [0 wt_one];
%     end
% end
