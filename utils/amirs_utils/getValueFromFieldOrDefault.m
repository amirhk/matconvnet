% -------------------------------------------------------------------------
function output = getValueFromFieldOrDefault(opts, field_string, default_value)
% -------------------------------------------------------------------------
  if isfield(opts, field_string) % i.e., if the field is set (has a value)
    output = opts.(field_string);
  else
    output = default_value;
  end
