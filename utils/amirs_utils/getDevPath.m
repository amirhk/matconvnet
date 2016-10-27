% --------------------------------------------------------------------
function datapath = getDevPath()
% --------------------------------------------------------------------
  if ispc
    datapath = 'H:\Amir';
  else
    datapath = '/Users/amirhk/dev';
  end
