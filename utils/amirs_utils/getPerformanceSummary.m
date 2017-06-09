% -------------------------------------------------------------------------
function [accuracy, sensitivity, specificity] = getPerformanceSummary(model_object, model_string, dataset, imdb, labels, set, should_return_summary)
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

  if isTwoClassImdb(dataset)
    fhGetAccSensSpec = @getAccSensSpec;
  else
    assert(isMultiClassImdb(dataset) || isSubsampledMultiClassImdb(dataset));
    fhGetAccSensSpec = @getAccSensSpecMultiClass;
  end
  if strcmp(set, 'train')
    set_number = 1;
  else
    set_number = 3;
  end
  
  keyboard
  if should_return_summary
    afprintf(sprintf('[INFO] Getting model performance on `%s` set...\n', set));
    [top_predictions, ~] = getPredictionsFromModelOnImdb(model_object, model_string, imdb, set_number);
    afprintf(sprintf('[INFO] Model performance on `%s` set\n', set));
    [ ...
      accuracy, ...
      sensitivity, ...
      specificity, ...
    ] = fhGetAccSensSpec(labels, top_predictions, true);
    printConsoleOutputSeparator();
  else
    accuracy = -1;
    sensitivity = -1;
    specificity = -1;
  end


