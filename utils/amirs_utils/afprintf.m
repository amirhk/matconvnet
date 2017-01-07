% --------------------------------------------------------------------
function afprintf(input_string, additional_tab_indent_count)
% --------------------------------------------------------------------
  if nargin == 1
    additional_tab_indent_count = 0;
  end
  [ST,~] = dbstack();
  stack_depth = numel(ST) - 1;
  indent_string = '';
  for i = 1:stack_depth + additional_tab_indent_count
    indent_string = strcat(indent_string, '\t');
  end
  % fprintf('%s%s', indent_string, input_string);
  fprintf([indent_string, input_string]);

