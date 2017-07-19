% -------------------------------------------------------------------------
function compact_array_string = getCompactStringRepresentationForNumbersArray(original_numbers_array)
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

  if length(original_numbers_array) == 0
    compact_array_string = 'empty_array';
  else
    compact_array_string = '';
    compact_array = [];
    compact_array(1) = original_numbers_array(1);
    counter = 1;
    for i = 2 : length(original_numbers_array)
      elem = original_numbers_array(i);

      if elem == compact_array(end)
        counter = counter + 1;
      else
        compact_array_string = strcat(compact_array_string, sprintf('\t%d x %.6f', counter, compact_array(end)));
        counter = 1;
        compact_array(end+1) = elem;
      end
    end
    compact_array_string = strcat(compact_array_string, sprintf('\t%d x %.6f ', counter, compact_array(end)));
    counter = 1;
    compact_array(end+1) = elem;
  end
