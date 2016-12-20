% -------------------------------------------------------------------------
function saveStruct2File(input_struct, filePath, recursion_depth)
% -------------------------------------------------------------------------
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
        fprintf(fileID, '%s: %d\n', fields{i}, value);
      case 'double'
        for j = 1:recursion_depth
          fprintf(fileID, '\t');
        end
        if numel(value) == 1
          fprintf(fileID, '%s: %3.2f\n', fields{i}, value);
        else
          fprintf(fileID, '%s: ', fields{i});
          for k = 1:numel(value)
            fprintf(fileID, '%3.2f  ',value(k));
          end
          fprintf(fileID, '\n');
        end

      case 'struct'
        for j = 1:recursion_depth
          fprintf(fileID, '\t');
        end
        fprintf(fileID, '%s:\n', fields{i});
        saveStruct2File(value, filePath, recursion_depth + 1)
        continue
    end
  end
