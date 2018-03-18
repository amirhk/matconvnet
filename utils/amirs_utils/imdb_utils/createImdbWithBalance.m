% --------------------------------------------------------------------
function imdb = createImdbWithBalance(dataset, imdb, train_balance_count, test_balance_count, should_save, debug_flag)
% --------------------------------------------------------------------
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



  fh_imdb_utils = imdbMultiClassUtils;
  % posneg_balance = sprintf('balanced-%d-%d', balance_count, balance_count);
  % afprintf(sprintf('[INFO] Constructing `%s`...\n', posneg_balance));
  if debug_flag; printConsoleOutputSeparator(); end;

  if debug_flag; afprintf(sprintf('[INFO] INITIAL IMDB INFO...\n')); end;
  if debug_flag; fh_imdb_utils.getImdbInfo(imdb, 1); end;
  imdb = fh_imdb_utils.balanceAllClassesInImdb(imdb, 'train', train_balance_count, debug_flag);
  imdb = fh_imdb_utils.balanceAllClassesInImdb(imdb, 'test', test_balance_count, debug_flag);
  if debug_flag; afprintf(sprintf('[INFO] FINAL IMDB INFO...\n')); end;
  if debug_flag; fh_imdb_utils.getImdbInfo(imdb, 1); end;
  if should_save
    % Save
    fh_imdb_utils.saveImdb(dataset, imdb, train_balance_count, test_balance_count);
  end
  if debug_flag; afprintf(sprintf('done!\n\n')); end;

















