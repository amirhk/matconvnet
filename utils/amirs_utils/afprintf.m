% --------------------------------------------------------------------
function afprintf(input_string, optional_subtract_tab_count)
% --------------------------------------------------------------------
  if nargin == 1
    optional_subtract_tab_count = 0;
  end
  [ST,~] = dbstack();
  stack_depth = numel(ST) - 1;
  indent_string = '';
  for i = 1:stack_depth - optional_subtract_tab_count
    indent_string = strcat(indent_string, '\t');
  end
  % fprintf('%s%s', indent_string, input_string);
  fprintf([indent_string, input_string]);

