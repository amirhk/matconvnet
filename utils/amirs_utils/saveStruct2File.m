% -------------------------------------------------------------------------
function saveStruct2File(input_struct, filePath, recursion_depth)
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

  % fileID = fopen(fullfile(expDir, 'sensitivity_specificity.txt'), 'w');
  format shortG
  fileID = fopen(filePath, 'a');
  fields = fieldnames(input_struct);
  for i = 1:numel(fields)
    value = input_struct.(fields{i});
    switch class(value)
      case 'char'
        for j = 1:recursion_depth
          fprintf(fileID, '\t');
        end
        fprintf(fileID, '%s: %s\n', fields{i}, value);
      case 'logical'
        for j = 1:recursion_depth
          fprintf(fileID, '\t');
        end
        if value
          fprintf(fileID, '%s: true\n', fields{i});
        else
          fprintf(fileID, '%s: false\n', fields{i});
        end
      case 'single'
        for j = 1:recursion_depth
          fprintf(fileID, '\t');
        end
        if numel(value) == 1
          if round(value) == value && value > 1
            fprintf(fileID, '%s: %d\n', fields{i}, value);
          else
            fprintf(fileID, '%s: %.6f\n', fields{i}, value);
          end
        else
          fprintf(fileID, '%s: ', fields{i});
          for k = 1:numel(value)
            if round(value(k)) == value(k) && value(k) > 1
              fprintf(fileID, '%d  ', value(k));
            else
              fprintf(fileID, '%.6f  ', value(k));
            end
          end
          fprintf(fileID, '\n');
        end
      case 'double'
        for j = 1:recursion_depth
          fprintf(fileID, '\t');
        end
        if numel(value) == 1
          if round(value) == value && value > 1
            fprintf(fileID, '%s: %d\n', fields{i}, value);
          else
            fprintf(fileID, '%s: %.6f\n', fields{i}, value);
          end
        else
          fprintf(fileID, '%s: ', fields{i});
          for k = 1:numel(value)
            if round(value(k)) == value(k) && value(k) > 1
              fprintf(fileID, '%8.d  ', value(k));
            else
              fprintf(fileID, '%.6f  ', value(k));
            end
          end
          fprintf(fileID, '\n');
        end
      case 'struct'
        for j = 1:recursion_depth
          fprintf(fileID, '\t');
        end
        fprintf(fileID, '%s:\n', fields{i});
        saveStruct2File(value, filePath, recursion_depth + 1);
      case 'cell'
        for j = 1:recursion_depth
          fprintf(fileID, '\t');
        end
        fprintf(fileID, '%s:\n', fields{i});
        % assuming the cell contains either all strings, or all doubles
        for x = value
          switch class(x)
            case 'char'
              fprintf(fileID, '%s  ', x);
            case 'double'
              if round(x) == x && x > 1
                fprintf(fileID, '%d  ', x);
              else
                fprintf(fileID, '%.6f  ', x);
              end
          end
        end
    end
  end
  fclose(fileID);
