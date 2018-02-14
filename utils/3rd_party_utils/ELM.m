% function [training_time, testing_time, training_accuracy, testing_accuracy] = elm(training_data_file, testing_data_file, elm_type, number_of_hidden_neurons, activation_function)
function [training_time, testing_time, training_accuracy, testing_accuracy, final_layer_projected_imdb] = elm(elm_type, activation_function, random_projection_type, dim_multiplier)

% Usage: elm(training_data_file, testing_data_file, elm_type, number_of_hidden_neurons, activation_function)
% OR:    [training_time, testing_time, training_accuracy, testing_accuracy] = elm(training_data_file, testing_data_file, elm_type, number_of_hidden_neurons, activation_function)
%
% Input:
% training_data_file     - Filename of training data set
% testing_data_file      - Filename of testing data set
% elm_type              - 0 for regression; 1 for (both binary and multi-classes) classification
% number_of_hidden_neurons - Number of hidden neurons assigned to the ELM
% activation_function    - Type of activation function:
%                           'sig' for Sigmoidal function
%                           'sin' for Sine function
%                           'hardlim' for Hardlim function
%                           'tribas' for Triangular basis function
%                           'radbas' for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
%
% Output:
% training_time          - Time (seconds) spent on training ELM
% testing_time           - Time (seconds) spent on predicting ALL testing data
% training_accuracy      - Training accuracy:
%                           RMSE for regression or correct classification rate for classification
% testing_accuracy       - Testing accuracy:
%                           RMSE for regression or correct classification rate for classification
%
% MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
% FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
% neurons; neuron 5 has the highest output means input belongs to 5-th class
%
% Sample1 regression: [training_time, testing_time, training_accuracy, testing_accuracy] = elm('sinc_train', 'sinc_test', 0, 20, 'sig')
% Sample2 classification: elm('diabetes_train', 'diabetes_test', 1, 20, 'sig')
%
    %%%%    Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
    %%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    %%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
    %%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
    %%%%    DATE:       APRIL 2004

  %%%%%%%%%%% Macro definition
  REGRESSION = 0;
  CLASSIFIER = 1;

  imdb = setupExperimentsUsingProjectedImbds('mnist-fashion-multi-class-subsampled', 'balanced-250', false, false);
  normalized_imdb = getNormalized4DImdbFrom4DImdb(imdb);
  vectorized_imdb = getVectorizedImdb(normalized_imdb);


  % -----------------------------------------------------------------------------
  %                                                              Project the imdb
  % -----------------------------------------------------------------------------

  % dim_multiplier = 1;
  if strcmp(random_projection_type, 'simple')
    % number_of_hidden_neurons = 49 * dim_multiplier;
    number_of_hidden_neurons = 100 * dim_multiplier;
    % number_of_hidden_neurons = 16 * dim_multiplier;
    % number_of_hidden_neurons = 4 * dim_multiplier;
    projected_vectorized_imdb = projectThroughSimpleRandomLayer(normalized_imdb, number_of_hidden_neurons, activation_function);
  elseif strcmp(random_projection_type, 'conv')
    number_of_hidden_kernels = 1 * dim_multiplier;
    projected_vectorized_imdb = projectThroughConvolutionalRandomLayer(normalized_imdb, number_of_hidden_kernels, activation_function);
  end


  % -----------------------------------------------------------------------------
  %                                                                   Other stuff
  % -----------------------------------------------------------------------------

  train_set_indices = projected_vectorized_imdb.images.set == 1;
  test_set_indices = projected_vectorized_imdb.images.set == 3;

  P = vectorized_imdb.images.data(train_set_indices,:)';
  TV.P = vectorized_imdb.images.data(test_set_indices,:)';

  h_train = projected_vectorized_imdb.images.data(train_set_indices,:)';
  h_test = projected_vectorized_imdb.images.data(test_set_indices,:)';

  T = projected_vectorized_imdb.images.labels(train_set_indices);
  TV.T = projected_vectorized_imdb.images.labels(test_set_indices);

  fprintf('\t%s-RP to %d-D\t', random_projection_type, size(h_train, 1));

  number_of_training_data = size(P,2);
  number_of_testing_data = size(TV.P,2);
  number_of_input_neurons = size(P,1);

  % -----------------------------------------------------------------------------
  %                                                            For Classification
  % -----------------------------------------------------------------------------

  if elm_type ~= REGRESSION
      %%%%%%%%%%%% Preprocessing the data of classification
      sorted_target=sort(cat(2,T,TV.T),2);
      label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
      label(1,1)=sorted_target(1,1);
      j=1;
      for i = 2:(number_of_training_data + number_of_testing_data)
          if sorted_target(1,i) ~= label(1,j)
              j=j+1;
              label(1,j) = sorted_target(1,i);
          end
      end
      number_class=j;
      number_of_output_neurons=number_class;

      %%%%%%%%%% Processing the targets of training
      temp_T=zeros(number_of_output_neurons, number_of_training_data);
      for i = 1:number_of_training_data
          for j = 1:number_class
              if label(1,j) == T(1,i)
                  break;
              end
          end
          temp_T(j,i)=1;
      end
      T=temp_T*2-1;

      %%%%%%%%%% Processing the targets of testing
      temp_TV_T=zeros(number_of_output_neurons, number_of_testing_data);
      for i = 1:number_of_testing_data
          for j = 1:number_class
              if label(1,j) == TV.T(1,i)
                  break;
              end
          end
          temp_TV_T(j,i)=1;
      end
      TV.T=temp_TV_T*2-1;
  end


  % -----------------------------------------------------------------------------
  %                               Calculate output weights output_weight (beta_i)
  % -----------------------------------------------------------------------------

  %%%%%%%%%%% Calculate weights & biases
  start_time_train = cputime;

  output_weight = pinv(h_train') * T';                                               % implementation without regularization factor //refer to 2006 Neurocomputing paper
  % output_weight = inv(eye(size(h_train,1))/C+h_train * h_train') * h_train * T';   % faster method 1 //refer to 2012 IEEE TSMC-B paper
                                                                                     % implementation; one can set regularizaiton factor C properly in classification applications
  % output_weight = (eye(size(h_train,1))/C+h_train * h_train') \ h_train * T';      % faster method 2 //refer to 2012 IEEE TSMC-B paper
                                                                                     % implementation; one can set regularizaiton factor C properly in classification applications

  % If you use faster methods or kernel method, PLEASE CITE in your paper properly:

  % Guang-Bin Huang, Hongming Zhou, Xiaojian Ding, and Rui Zhang, "Extreme Learning Machine for Regression and Multi-Class Classification," submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence, October 2010.

  end_time_train = cputime;
  training_time = end_time_train - start_time_train;      %   Calculate CPU time (seconds) spent for training ELM


  % -----------------------------------------------------------------------------
  %                                               Calculate the training accuracy
  % -----------------------------------------------------------------------------

  Y = (h_train' * output_weight)';                        %   Y: the actual output of the training data
  clear h_train;


  % -----------------------------------------------------------------------------
  %                                         Calculate the output of testing input
  % -----------------------------------------------------------------------------

  start_time_test = cputime;
  TY = (h_test' * output_weight)';                        %   TY: the actual output of the testing data
  end_time_test = cputime;
  testing_time = end_time_test - start_time_test;         %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data




  % -----------------------------------------------------------------------------
  %                              Calculate training & testing regression accuracy
  % -----------------------------------------------------------------------------

  if elm_type == REGRESSION
      training_accuracy = sqrt(mse(T - Y));               %   Calculate training accuracy (RMSE) for regression case
      testing_accuracy = sqrt(mse(TV.T - TY));            %   Calculate testing accuracy (RMSE) for regression case
  end


  % -----------------------------------------------------------------------------
  %                          Calculate training & testing classification accuracy
  % -----------------------------------------------------------------------------

  if elm_type == CLASSIFIER
      miss_classification_rate_training = 0;
      miss_classification_rate_testing = 0;

      for i = 1 : size(T, 2)
          [x, label_index_expected] = max(T(:,i));
          [x, label_index_actual] = max(Y(:,i));
          if label_index_actual ~= label_index_expected
              miss_classification_rate_training = miss_classification_rate_training + 1;
          end
      end
      training_accuracy = 1 - miss_classification_rate_training / size(T, 2);

      for i = 1 : size(TV.T, 2)
          [x, label_index_expected] = max(TV.T(:,i));
          [x, label_index_actual] = max(TY(:,i));
          if label_index_actual ~= label_index_expected
              miss_classification_rate_testing = miss_classification_rate_testing + 1;
          end
      end
      testing_accuracy = 1 - miss_classification_rate_testing / size(TV.T, 2);
  end

  projected_data_train = Y';
  projected_data_test = TY';
  number_of_train_samples = size(projected_data_train, 1);
  number_of_test_samples = size(projected_data_test, 1);

  [~, projected_labels_train] = sort(T, 1, 'descend');
  [~, projected_labels_test] = sort(TV.T, 1, 'descend');
  projected_labels_train = projected_labels_train(1,:)'; % get the first row, which contains the highest value in each column
  projected_labels_test = projected_labels_test(1,:)'; % get the first row, which contains the highest value in each column

  projected_set_train = ones(number_of_train_samples, 1) * 1;
  projected_set_test = ones(number_of_test_samples, 1) * 3;

  projected_train_imdb.images.data = projected_data_train;
  projected_train_imdb.images.labels = projected_labels_train;
  projected_train_imdb.images.set = projected_set_train;

  projected_test_imdb.images.data = projected_data_test;
  projected_test_imdb.images.labels = projected_labels_test;
  projected_test_imdb.images.set = projected_set_test;

  projected_train_imdb = get4DImdb(projected_train_imdb, 10, 1, 1, number_of_train_samples);
  projected_test_imdb = get4DImdb(projected_test_imdb, 10, 1, 1, number_of_test_samples);

  % IMPORTANT: this is projected to the final layer... not the hidden layer
  final_layer_projected_imdb = mergeImdbs(projected_train_imdb, projected_test_imdb, false);


% --------------------------------------------------------------------
function projected_vectorized_imdb = projectThroughSimpleRandomLayer(normalized_imdb, number_of_hidden_neurons, activation_function)
% --------------------------------------------------------------------

  vectorized_imdb = getVectorizedImdb(normalized_imdb);
  number_of_input_neurons = size(vectorized_imdb.images.data, 2);
  input_weight = rand(number_of_hidden_neurons, number_of_input_neurons) * 2 - 1;
  bias_of_hidden_neurons = rand(number_of_hidden_neurons, 1);

  projected_data = helperSimpleRandomProjection(input_weight, bias_of_hidden_neurons, activation_function, vectorized_imdb.images.data');
  projected_vectorized_imdb = vectorized_imdb;
  projected_vectorized_imdb.images.data = projected_data';

  % h_train = helperSimpleRandomProjection(input_weight, bias_of_hidden_neurons, activation_function, data_train);
  % h_test = helperSimpleRandomProjection(input_weight, bias_of_hidden_neurons, activation_function, data_test);

% --------------------------------------------------------------------
function H = helperSimpleRandomProjection(input_weight, bias_of_hidden_neurons, activation_function, data)
% --------------------------------------------------------------------
  tempH = input_weight * data;
  ind = ones(1, size(data, 2));
  % Removed the two lines below, as this doens't really need to be here for comparison w/ conv-RP
  % BiasMatrix = bias_of_hidden_neurons(:,ind);
  % tempH = tempH + BiasMatrix;
  switch lower(activation_function)
      case {'sig','sigmoid'}
          %%%%%%%% Sigmoid
          H = 1 ./ (1 + exp(-tempH));
      case {'sin','sine'}
          %%%%%%%% Sine
          H = sin(tempH);
      case {'hardlim'}
          %%%%%%%% Hard Limit
          H = hardlim(tempH);
      case {'tribas'}
          %%%%%%%% Triangular basis function
          H = tribas(tempH);
      case {'radbas'}
          %%%%%%%% Radial basis function
          H = radbas(tempH);
          %%%%%%%% More activation functions can be added here
  end

% --------------------------------------------------------------------
function projected_vectorized_imdb = projectThroughConvolutionalRandomLayer(normalized_imdb, number_of_hidden_kernels, activation_function)
% --------------------------------------------------------------------

  % For 1 layer,  w/ stride 4, N kernels result in ( 49 x N)-D output (layer 1: 28^2 -> 7^2)
  % For 1 layer,  w/ stride 3, N kernels result in (100 x N)-D output (layer 1: 28^2 -> 10^2)
  % For 2 layers, w/ stride 3, N kernels result in ( 16 x N)-D output (layer 1: 28^2 -> 10^2, layer 2: 10^2 -> 4^2)
  % For 3 layers, w/ stride 3, N kernels result in (  4 x N)-D output (layer 1: 28^2 -> 10^2, layer 2: 10^2 -> 4^2, layer 3: 4^2 -> 2^2)
  projection_string = sprintf('custom-1-L-5-99-%d-%s', number_of_hidden_kernels, activation_function);
  % projection_string = sprintf('custom-2-L-5-64-%d-%s', number_of_hidden_kernels, activation_function);
  % projection_string = sprintf('custom-3-L-5-64-%d-%s', number_of_hidden_kernels, activation_function);

  larp_weight_init_type = 'random-between-pm-one'; % 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  projected_imdb = getProjectedImdbUsingMatConvNet(normalized_imdb, 'mnist-fashion', larp_weight_init_type, projection_string, -1);
  % IMPORTANT: I'm projecting the entire IMDB on one network (same weights)
  vectorized_projected_imdb = getVectorizedImdb(projected_imdb);
  projected_vectorized_imdb = vectorized_projected_imdb;

  % train_set_indices = vectorized_projected_imdb.images.set == 1;
  % test_set_indices = vectorized_projected_imdb.images.set == 3;

  % h_train = vectorized_projected_imdb.images.data(train_set_indices,:)';
  % h_test = vectorized_projected_imdb.images.data(test_set_indices,:)';

% -------------------------------------------------------------------------
function projected_imdb = getProjectedImdbUsingMatConvNet(imdb, dataset, larp_weight_init_type, larp_network_arch, projection_depth)
% -------------------------------------------------------------------------
  fh_projection_utils = projectionUtils;
  larp_weight_init_sequence = getLarpWeightInitSequence(larp_weight_init_type, larp_network_arch);
  projection_net = fh_projection_utils.getProjectionNetworkObject(dataset, larp_network_arch, larp_weight_init_sequence);
  projected_imdb = fh_projection_utils.projectImdbThroughNetwork(imdb, projection_net, projection_depth);
















