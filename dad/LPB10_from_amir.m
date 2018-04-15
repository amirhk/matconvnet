% -------------------------------------------------------------------------
function LPB_dad()
% -------------------------------------------------------------------------
% Copyright (c) 2018, Gholamreza Karimi, Amir-Hossein Karimi
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

    % Laser Powder Bed Additive Manufacturing
    % clear all;
    % close all;
    % clc;


    %DESIGN & OPERATING CONDITIONS
    Time = 1;
    Dt = 1;

    % Ambient Conditions
    Tini_scalar = 20;                                % Initial temperature [C]
    Tamb_scalar = 20;                                % Ambient temperature [C]
    hc = 20;                                  % Convective heat transfer coefficint [W/m^2.C]
    Sigma = 5.669e-8;                         % Estefan-Boltzmann Const. W/m^2.K^4

    % LASER
    P_L = 200;                                % Laser power [W]
    R_L = 1e-3;                               % Laser spot diamter (on the powder surface) [m^2]
    V_L = 4e-3;                               % Laser scanning velocity [m/s]

    % MELT
    Beta = 0.35;                              % Surface absrbtivity
    % Tmelt = 1400;                             % Powder melting point [C]
    Tmelt = 140;                             % Powder melting point [C]
    Tboil = 7000;                             % Boiling temperature [C]
    Lambda_SL = 204500;                       % Powder latent heat of fusion [J/kg
    Lambda_LV = Lambda_SL;                    % CHANGE THIS LATER


    % DIMENSIONS (Laser beam position initially at the intesections of Lw, Le, Ln, Ls and at z = 0)
    Lw = 4 * R_L;
    Le = 4 * R_L + V_L * Time;
    % Le = 4 * R_L;
    Ln = 4 * R_L;
    Ls = 4 * R_L;

    Length = Le + Lw;
    Width = Ln + Ls;
    Height = 3e-3;

    % BED PROPERTIES
    Rho_P = 8440;                              % powder density [kg/m^3]
    Rho_A = 1;                                % Inert gas density [kg/m^3]
    Cp_P = 410;                               % Powder specific heat [J/kg.C]
    Cp_A = 1000;                              % Inert gas specific heat [J/kg.C]
    k_P = 10;                                 % Powder thermal conductivity [W/m.C]
    k_A = 0.02;                               % Inert gas thermal conductivity [W/m.C]
    Phi = 0.5;                                % Powder porosoity [%]

    % BULK (POWDER + INERT GAS) PROPERTIES
    Rho_B = Rho_P * (1-Phi) + Rho_A * Phi;    % Bulk density [kg/m^3]
    Cp_B = Cp_P * (1-Phi) + Cp_A * Phi;       % Bulk specific [J/kg.C]
    k_B = k_P * (1-Phi) + k_A * Phi;          % Bulk thermal conductivity [W/m.C]
    Alfa_B = k_B / (Rho_B * Cp_B);            % Bulk thermal diffusivity [m^2/s]

    % No of increments in each direction
    % nx = 10; % 20;
    % ny = 10; % 14;
    % nz = 3; % 10;
    nx = 2 * round(Length / R_L);
    ny = 2 * round(Width / R_L);
    nz = 4;

    Dx = Length / nx;
    Dy = Width / ny;
    Dz = Height / nz;

    % Total No of nodes in the domain
    Nx = nx + 1;
    Ny = ny + 1;
    Nz = nz + 1;

    x_bins = linspace(-Lw, Le, Nx);
    y_bins = linspace(-Ls, Ln, Ny);

    A_L = pi * R_L^2;                           % Laser spot area [m^2]

    % INITIALIZATION (TEMPERATURES, HEAT SOURCES, AND STATE OF PHASE)
    T_ini = Tini_scalar * ones(Nx, Ny, Nz);
    T_amb = Tamb_scalar * ones(Nx, Ny, Nz);
    T_Gauss = T_ini;

    Sum_E1 = 0;   % Stored energy - Rosenthal's equation
    Sum_E2 = 0;   % Stored energy - Gaussian
    Sum_E3 = 0;   % Heat dissipation - Convection
    Sum_E4 = 0;   % Heat dissipation - Radiation

    x_L_ini = 0;
    y_L_ini = 0;
    z_L_ini = 0;

    %x_L_fin = x_L_ini + V_L * Dt;
    y_L_fin = 0;
    z_L_fin = 0;

    % SIMULATION STARTS HERE
    t_Process = 1;
    N_Dt = 1;
    Dt = t_Process / N_Dt;

    scale_factor_alpha = (nx + 1 - 1) / Length;
    scale_factor_beta = (ny + 1 - 1) / Width;

    for t = 1:2,

        counter = 1;
        total_count = Nz * Ny * Nx;

        Q_Conv(:,:) = hc * (T_Gauss(:,:,1) - T_amb(:,:,1));
        Q_Rad(:,:) = Sigma * Beta * ((T_Gauss(:,:,1) + 273.15).^4 - (T_amb(:,:,1)+273.15).^4);
        Q_melt(:,:) = sum(T_Gauss > Tmelt, 3) * Lambda_SL * Rho_B * Dz;
        Q_Comb(:,:) = Q_Conv + Q_Rad + Q_melt;
        if t == 1
            params = [0,0,0,0,0];
        else
            params = fit2DGaussianToQ(Q_Comb', 2 * P_L * Beta / (pi * R_L^2), 'local', x_bins, y_bins)
        end

        for k = 1:Nz,
            for j = 1:Ny,
                for i = 1:Nx,

                    x = (i-1) * Dx - Lw;
                    y = (j-1) * Dy - Ls;
                    z = (k-1) * Dz;

                    % Instantaneous laser location
                    % Kisi = x - x_L_fin;
                    x_L_fin = x_L_ini + V_L * Dt;
                    Kisi = x - x_L_fin;
                    Eta = y - y_L_fin;
                    Zeta = z - z_L_fin;

                    R_Ros = sqrt(Kisi^2 + Eta^2 + Zeta^2);

                    % if R_Ros < 1e-9,
                    %   R_Ros=Dz/10;
                    % end;

                    % ------------------------------------------------------------------
                    % Energy flux / distribution delivered to bed. (Watt / m^2)
                    % ------------------------------------------------------------------
                    II(i,j) = 2 * P_L * Beta / (pi * R_L^2) * exp(-2 * (Kisi^2 + y^2) / R_L^2);

                    % ------------------------------------------------------------------
                    % T Rosenthal
                    % ------------------------------------------------------------------
                    T_Ros(i,j,k) = T_ini(i,j,k) + P_L * Beta / (2 * pi * k_B * R_Ros) * exp(- V_L * (Kisi + R_Ros) / (2 * Alfa_B));

                    % ------------------------------------------------------------------
                    % T Gauss
                    % ------------------------------------------------------------------
                    constant = 1;
                    fun_R = @(alpha_, beta_) sqrt( (x - (x_L_fin + alpha_)).^2 + (y - beta_).^2 + z.^2);
                    % fun_Q = @(alpha_, beta_) 10e6;
                    fun_Q = @(alpha_, beta_) params(1) * exp( -((alpha_ - params(2)) .^ 2 / (2 * params(3) ^ 2) + (beta_ - params(4)) .^ 2 / (2 * params(5) ^ 2)) );
                    fun_I = @(alpha_, beta_) 2 * P_L * Beta / (pi * R_L^2) .* exp(-2 .* (alpha_.^2 + beta_.^2) / R_L.^2);
                    fun_I_modified = @(alpha_, beta_) (fun_I(alpha_, beta_) - fun_Q(alpha_, beta_));
                    fun_kolli = @(alpha_, beta_) ...
                        fun_I_modified(alpha_, beta_) .* ...
                        (1 / (2 * pi * k_B)) ./ fun_R(alpha_, beta_) .* exp(- V_L .* (x - (x_L_fin + alpha_) + fun_R(alpha_, beta_)) / (2*Alfa_B));

                    % IMPORTANT: The bounds below work because we're assuming the laser starts at the origin!
                    %            So we're taking integral in a circle of radius R_L around the origin...
                    %            hmmm... but how does this consider the final location of the laser???
                    alpha_min = - R_L;
                    alpha_max = R_L;
                    beta_min = @(alpha_) - sqrt(R_L.^2 - alpha_.^2);
                    beta_max = @(alpha_) + sqrt(R_L.^2 - alpha_.^2);
                    % alpha_min = - Lw;
                    % alpha_max = Le;
                    % beta_min = -Ls;
                    % beta_max = Ln;

                    T_Gauss(i,j,k) = T_ini(i,j,k) + constant * integral2(fun_kolli, alpha_min, alpha_max, beta_min, beta_max);

                    % ------------------------------------------------------------------
                    % Others....
                    % ------------------------------------------------------------------
                    % if T_Gauss(i,j,k) > Tmelt,
                    %     Qsens = Rho_B * Dx * Dy * Dz * Cp_B * (T_Gauss(i,j,k) - Tmelt);
                    %     Qlat = Rho_B * Dx * Dy * Dz * Lambda_SL;
                    %     T_Gauss(i,j,k) = Tmelt + (Qsens - Qlat) /(Rho_B * Dx * Dy * Dz * Cp_B);
                    %       if T_Gauss(i,j,k) > Tboil,
                    %           Qsens = Rho_B * Dx * Dy * Dz * Cp_B * (T_Gauss(i,j,k) - Tboil);
                    %           Qlat = Rho_B * Dx * Dy * Dz * Lambda_LV;
                    %           T_Gauss(i,j,k) = Tboil + (Qsens - Qlat) /(Rho_B * Dx * Dy * Dz * Cp_B);
                    %         end;
                    % end;

                    % Sum_E1 = Sum_E1 + Rho_B * Cp_B * Dx * Dy * Dz * (T_Ros(i,j,k) - Tini_scalar) * Dt;
                    % Sum_E2 = Sum_E2 + Rho_B * Cp_B * Dx * Dy * Dz * (T_Gauss(i,j,k) - Tini_scalar) * Dt;
                    % fprintf('Iteration #%03d/%03d \t \t %.3f \t %.3f \t \t %.3f \t \t %.3f\n', counter, total_count, t, Sum_E1, Sum_E2, Sum_E1/Sum_E2);
                    % [Sum_E1 Sum_E2 Sum_E1/Sum_E2];

                    fprintf('Iteration #%03d/%03d\n', counter, total_count);
                    counter = counter + 1;
                end
            end
        end

        % T = T_Gauss;
        % x_L_ini = x_L_fin;

    end

    [xx, yy] = meshgrid(x_bins, y_bins);
    figure,
    subplot(2,2,1);
    surfc(xx, yy, T_Ros(:,:,1)');
    colorbar;
    xlabel('x [m]');
    ylabel('y [m]');
    zlabel('T_{Ros1} [C]');
    title('T_{Ros1} @ layer 1');
    pbaspect([Length / Width 1 1]);

    subplot(2,2,2);
    surfc(xx, yy, T_Gauss(:,:,1)');
    colorbar
    xlabel('x [m]');
    ylabel('y [m]');
    zlabel('T_{Gauss} [C]');
    title('T_{Gauss} @ layer 1');
    pbaspect([Length / Width 1 1]);

    subplot(2,2,3)
    surfc(xx, yy, T_Ros(:,:,2)');
    colorbar
    xlabel('x');
    ylabel('y');
    zlabel('T_{Ros2} [C]');
    title('T_{Ros2} @ layer 2');
    pbaspect([Length / Width 1 1]);

    subplot(2,2,4)
    surfc(xx, yy, T_Gauss(:,:,2)');
    colorbar
    xlabel('x [m]');
    ylabel('y [m]');
    zlabel('T_{Gauss} [C]');
    title('T_{Gauss} @ layer 2');
    pbaspect([Length / Width 1 1]);

    % size(tmp_1) = [];
    % size(tmp_2) = [];
    % size(tmp_3) = [];
    % i = 1;
    % for x = unique(xx(1,:))
    %     j = 1;
    %     for y = unique(yy(:,:))'
    %         tmp_1(j,i) = fun_Q(x * scale_factor_alpha, y * scale_factor_beta);
    %         tmp_2(j,i) = fun_I(x,y);
    %         tmp_3(j,i) = fun_I_modified(x,y);
    %         j = j + 1;
    %     end
    %     i = i + 1;
    % end

    keyboard

    % tmp_1 = fun_Q(xx * scale_factor_alpha, yy * scale_factor_beta)';
    tmp_1 = fun_Q(xx, yy);
    tmp_2 = fun_I(xx, yy);
    tmp_3 = fun_I_modified(xx, yy);

    figure,
    subplot(2,2,1), surf(xx, yy, Q_Comb'), title('Q Comb'),     % zlim([-5e6,5e6]),
    subplot(2,2,2), surf(xx, yy, tmp_1),   title('Q Fit'),      % zlim([-5e6,5e6]),
    subplot(2,2,3), surf(xx, yy, tmp_2),   title('I'),          % zlim([-1e7,5e7]),
    subplot(2,2,4), surf(xx, yy, tmp_3),   title('I modified'), % zlim([-1e7,5e7]),




% --------------------------------------------------------------------
function params = fit2DGaussianToQ(Q_original, initial_guess_magnitude, fit_type, x_bins, y_bins)
% --------------------------------------------------------------------

    % IMPORTANT: Q_original should be as follows
    %
    %  <--           Lw + Le           -->
    %  ___________________________________  ^
    % |                                   | |
    % |                                   | |
    % |                                   |
    % |                                   | Ln + Ls
    % |                                   |
    % |                                   | |
    % |                                   | |
    % |___________________________________| v
    %

    assert(size(Q_original, 1) == numel(y_bins));
    assert(size(Q_original, 2) == numel(x_bins));

    Q_local = Q_original;
    Q_local(Q_local < 0.4 * max(Q_local(:))) = 0;
    drop_ratio = sum(Q_local(:)) / sum(Q_original(:));
    if isnan(drop_ratio) % for first iteration when Q_original is all zeros
        drop_ratio = 1;
    end

    initial_guess_magnitude = max(Q_original(:));
    [row_max, col_max] = find(ismember(Q_original, max(Q_original(:))));
    initial_guess_x = x_bins(col_max);
    initial_guess_y = y_bins(row_max);

    [params_all, Q_fit_all] = fit2DGaussianToMatrix(Q_original, initial_guess_magnitude, initial_guess_x, initial_guess_y, x_bins, y_bins);
    [params_local, Q_fit_local] = fit2DGaussianToMatrix(Q_local, initial_guess_magnitude, initial_guess_x, initial_guess_y, x_bins, y_bins);

    [xx, yy] = meshgrid(x_bins, y_bins);
    figure,
    subplot(1,3,1), surf(xx, yy, Q_original), title('Q Comb')
    subplot(1,3,2), surf(xx, yy, Q_fit_all), title('Q fit all')
    subplot(1,3,3), surf(xx, yy, Q_fit_local), title('Q fit local')
    suptitle('Fitting Process')

    if strcmp(fit_type, 'all')
        params = params_all;
    elseif strcmp(fit_type, 'local')
        params = params_local;
    end


% --------------------------------------------------------------------
function [params, S_fit] = fit2DGaussianToMatrix(S, initial_guess_magnitude, initial_guess_x, initial_guess_y, x_bins, y_bins);
% --------------------------------------------------------------------
    distance_between_x_bins = x_bins(2) - x_bins(1);
    distance_between_y_bins = y_bins(2) - y_bins(1);
    A0 = [ ... % Initial (guess) parameters
        initial_guess_magnitude, ...
        initial_guess_x, ...
        distance_between_x_bins, ...
        initial_guess_y, ...
        distance_between_y_bins];

    lb = [0, min(x_bins), 0, min(y_bins), 0];
    ub = [realmax('double'), max(x_bins), sqrt(max(x_bins)), max(y_bins), sqrt(max(y_bins))];

    [xx, yy] = meshgrid(x_bins, y_bins);
    X = zeros(numel(y_bins), numel(x_bins), 2);
    X(:, :, 1) = xx;
    X(:, :, 2) = yy;

    g = @(A,X) A(1) * exp( -((X(:,:,1)-A(2)).^2 / (2 * A(3)^2) + (X(:,:,2) - A(4)).^2 / (2 * A(5)^2)) );

    options = optimoptions('lsqcurvefit','MaxFunctionEvaluations', 10000, 'MaxIterations', 10000);
    [A, resnorm, res, flag, output] = lsqcurvefit(g, A0, X, S, lb, ub, options);
    params = A;
    params(1)
    params(2:end)

    S_fit = g(params, X);









% % --------------------------------------------------------------------
% function [params, S_fit] = fit2DGaussianToMatrix(S, initial_guess_magnitude, initial_guess_x, initial_guess_y, x_bins, y_bins);
% % --------------------------------------------------------------------
%     n = size(S, 2) - 1;
%     m = size(S, 1) - 1;
%     A0 = [initial_guess_magnitude, initial_guess_x, .01, initial_guess_y, .01, 0]; % Initial (guess) parameters

%     lb = [0, -n/2, 0, -m/2, 0, 0];
%     ub = [realmax('double'), n/2, (n/2)^2, m/2, (m/2)^2, pi/4];

%     % keyboard
%     % size(meshgrid(y_bins, x_bins))
%     % size(meshgrid(-n/2 : n/2, -m/2 : m/2))
%     % [x, y] = meshgrid(-n/2 : n/2, -m/2 : m/2);
%     [x, y] = meshgrid(y_bins, x_bins);
%     X = zeros(m + 1, n + 1, 2); % +1 because of 0 in between -(n,m)/2 and +(n,m)/2
%     X(:, :, 1) = x;
%     X(:, :, 2) = y;

%     g = @(A,X) A(1) * exp( -((X(:,:,1)-A(2)).^2 / (2 * A(3)^2) + (X(:,:,2) - A(4)).^2 / (2 * A(5)^2)) );
%     % f = @(A,X) A(1)*exp( -(...
%     %     ( X(:,:,1)*cos(A(6))-X(:,:,2)*sin(A(6)) - A(2)*cos(A(6))+A(4)*sin(A(6)) ).^2/(2*A(3)^2) + ...
%     %     ( X(:,:,1)*sin(A(6))+X(:,:,2)*cos(A(6)) - A(2)*sin(A(6))-A(4)*cos(A(6)) ).^2/(2*A(5)^2) ) );

%     h = g;
%     % h = f;

%     options = optimoptions('lsqcurvefit','MaxFunctionEvaluations',10000, 'MaxIterations',10000);
%     [A, resnorm, res, flag, output] = lsqcurvefit(h, A0, X, S, lb, ub, options);
%     params = A;
%     params(1)
%     params(2:end)

%     S_fit = h(params, X);
%     % keyboard

% % --------------------------------------------------------------------
% function [params, S_fit] = fit2DGaussianToMatrix2(S, initial_guess_magnitude)
% % --------------------------------------------------------------------
%     n = size(S, 2) - 1;
%     m = size(S, 1) - 1;
%     A0 = [initial_guess_magnitude, 0, 1, 0, 1, 0]; % Initial (guess) parameters

%     lb = [0,-n/2,0,-n/2,0,0];
%     ub = [realmax('double'),n/2,(n/2)^2,n/2,(n/2)^2,pi/4];

%     [x,y] = meshgrid(-n/2:n/2,-m/2:m/2);
%     X=zeros(m+1,n+1,2);
%     X(:,:,1) = x;
%     X(:,:,2) = y;

%     g = @(A,X) A(1)*exp( -((X(:,:,1)-A(2)).^2/(2*A(3)^2) + (X(:,:,2)-A(4)).^2/(2*A(5)^2)) );
%     f = @(A,X) A(1)*exp( -(...
%         ( X(:,:,1)*cos(A(6))-X(:,:,2)*sin(A(6)) - A(2)*cos(A(6))+A(4)*sin(A(6)) ).^2/(2*A(3)^2) + ...
%         ( X(:,:,1)*sin(A(6))+X(:,:,2)*cos(A(6)) - A(2)*sin(A(6))-A(4)*cos(A(6)) ).^2/(2*A(5)^2) ) );

%     h = g;
%     % h = f;

%     options = optimoptions('lsqcurvefit','MaxFunctionEvaluations',10000);
%     [A, resnorm, res, flag, output] = lsqcurvefit(h, A0, X, S, lb, ub, options);
%     params = A;
%     params(1)
%     params(2:end)

%     S_fit = h(params,X);
%     % keyboard






















