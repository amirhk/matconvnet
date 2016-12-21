% -------------------------------------------------------------------------
function struct_1 = mergeStructs(struct_1, struct_2)
% -------------------------------------------------------------------------
  f = fieldnames(struct_2);
  for i = 1:length(f)
    struct_1.(f{i}) = struct_2.(f{i});
  end
