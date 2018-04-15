% -------------------------------------------------------------------------
function LPB()
% -------------------------------------------------------------------------
% Copyright (c) 2018, Reza Karimi, Amir-Hossein Karimi
% All rights reserved.

    % Laser Powder Bed Additive Manufacturing
    clear all;
    close all;
    clc;


    %DESIGN & OPERATING CONDITIONS
    Time = 1;
    Dt = 1;

    % Ambient Conditions
    Tini_scalar = 20;
    Tamb_scalar = 20;
    hc = 20;
    Sigma = 5.669e-8;

    % LASER SPECIFICATIONS
    P_L = 100;
    R_L = 0.5e-3;
    V_L = 1e-3;

    % MELT
    Beta = 0.35;
    Tmelt = 1340;
    DTm = 100;
    Tboil = 7000;
    Lambda_SL = 204500;
    Lambda_LV = Lambda_SL;


    % DIMENSIONS
    Lw = 4 * R_L;
    % Le = V_L * Time + 4 * R_L;
    Le = 4 * R_L;
    Ln = 4 * R_L;
    Ls = 4 * R_L;

    Length = Le + Lw;
    Width = Ln + Ls;
    Height = 1e-3;

    % BED PROPERTIES
    Rho_P = 8440;
    Rho_A = 1;
    Cp_P = 410;
    Cp_A = 1000;
    k_P = 10;
    k_A = 0.02;
    Phi = 0.5;

    % BULK (POWDER + INERT GAS) PROPERTIES
    Rho_B = Rho_P * (1 - Phi) + Rho_A * Phi;
    Cp_B = Cp_P * (1 - Phi) + Cp_A * Phi;
    k_B = k_P * (1 - Phi) + k_A * Phi;
    Alfa_B = k_B / (Rho_B * Cp_B);

    % INCREMENTS
    nx = 2 * round(Length / R_L);
    ny = 2 * round(Width / R_L);
    nz = 5;

    Dx = Length / nx;
    Dy = Width / ny;
    Dz = Height / nz;

    Nx = nx + 1;
    Ny = ny + 1;
    Nz = nz + 1;

    A_L = pi * R_L^2;                           % Laser spot area [m^2]


    % INITIALIZATION (TEMPERATURES, HEAT SOURCES, AND STATE OF PHASE)
    T_ini = Tini_scalar * ones(Nx, Ny, Nz);
    T_amb = Tamb_scalar * ones(Nx, Ny, Nz);
    T_Gauss = T_ini;

    x_L_ini = 0;
    y_L_ini = 0;
    z_L_ini = 0;

    y_L_fin = 0;
    z_L_fin = 0;

    % SIMULATION STARTS HERE
    scale_factor_alpha = (nx + 1 - 1) / Length;
    scale_factor_beta = (ny + 1 - 1) / Width;

    N_Dt = Time / Dt;

    N_Itr = 2;

    for nt = 1 : N_Dt

        T_ini = Tini_scalar * ones(Nx, Ny, Nz);
        T_amb = Tamb_scalar * ones(Nx, Ny, Nz);
        T_Gauss = T_ini;

        for nn = 1 : N_Itr

            counter = 1;

            Sum_E1 = 0;
            Sum_E2 = 0;

            total_count = Nz * Ny * Nx;

            Q_Conv(:,:) = hc * (T_Gauss(:,:,1) - T_amb(:,:,1));
            Q_Rad(:,:) = Sigma * Beta * ((T_Gauss(:,:,1) + 273.15).^4 - (T_amb(:,:,1) + 273.15).^4);
            Q_Melt(:,:) = sum(T_Gauss > Tmelt, 3) * Lambda_SL * Rho_B * Dz;
            Q_Comb(:,:) = Q_Conv + Q_Rad + Q_Melt;

            fit_params = fit2DGaussianToQ(Q_Comb, 2 * P_L * Beta / (pi * R_L^2))

            for k = 1 : Nz
                for j = 1 : Ny
                    for i = 1 : Nx

                        x = (i-1) * Dx - Lw;
                        y = (j-1) * Dy - Ls;
                        z = (k-1) * Dz;

                        % Instantaneous laser location
                        x_L_fin = x_L_ini + V_L * Dt;

                        Kisi = x - x_L_fin;
                        Eta = y - y_L_fin;
                        Zeta = z - z_L_fin;

                        R_Ros = sqrt(Kisi^2 + Eta^2 + Zeta^2);

                        if R_Ros < 1e-9,
                            R_Ros=Dz/10;
                        end

                        % ------------------------------------------------------------------
                        % Energy flux / distribution delivered to bed. (Wat / m^2)
                        % ------------------------------------------------------------------
                        II(i,j) = 2 * P_L * Beta / (pi * R_L^2) * exp(-2 * (Kisi^2 + y^2) / R_L^2);

                        % ------------------------------------------------------------------
                        % T Rosenthal
                        % ------------------------------------------------------------------
                        T_Ros(i,j,k) = T_ini(i,j,k) + P_L * Beta / (2 * pi * k_B * R_Ros) * exp(- V_L * (Kisi + R_Ros) / (2 * Alfa_B));

                        % ------------------------------------------------------------------
                        % T Gauss
                        % ------------------------------------------------------------------
                        fun_R = @(alpha_, beta_) sqrt( (x - (x_L_fin + alpha_)).^2 + (y - beta_).^2 + z.^2);
                        % fun_Q = @(alpha_, beta_) 10e6;
                        fun_Q = @(alpha_, beta_) fit_params(1) * exp( -((alpha_ - fit_params(2)) .^ 2 / (2 * fit_params(3) ^ 2) + (beta_ - fit_params(4)) .^ 2 / (2 * fit_params(5) ^ 2)) );
                        fun_I = @(alpha_, beta_) 2 * P_L * Beta / (pi * R_L^2) .* exp(-2 .* (alpha_.^2 + beta_.^2) / R_L.^2);
                        fun_I_modified = @(alpha_, beta_) (fun_I(alpha_, beta_) - fun_Q(alpha_ * scale_factor_alpha, beta_ * scale_factor_beta));
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

                        T_Gauss(i,j,k) = T_ini(i,j,k) + integral2(fun_kolli,alpha_min,alpha_max,beta_min,beta_max);

                        % ------------------------------------------------------------------
                        % Others....
                        % ------------------------------------------------------------------
                        Sum_E1 = Sum_E1 + Rho_B * Cp_B * Dx * Dy * Dz * (T_Ros(i,j,k) - Tini_scalar);
                        Sum_E2 = Sum_E2 + Rho_B * Cp_B * Dx * Dy * Dz * (T_Gauss(i,j,k) - Tini_scalar);

                        fprintf( ...
                            'Iteration #%03d/%03d \t \t %.3f \t %.3f \t %.3f \t %.3f \t %.3f\n', ...
                            counter, ...
                            total_count, ...
                            nt, ...
                            nn, ...
                            (P_L * Beta * Time), Sum_E1, Sum_E2);
                        counter = counter + 1;

                    end
                end
            end

            % [xx, yy] = meshgrid(linspace(-Lw,Le,Nx),linspace(-Ls,Ln,Ny));
            % figure,
            % tmp = [];
            % i = 1;
            % for x = unique(xx(1,:))
            %     j = 1;
            %     for y = unique(yy(:,:))'
            %         tmp(j,i) = fun_I_modified(x,y);
            %         j = j + 1;
            %     end
            %     i = i + 1;
            % end
            % surf(xx,yy,tmp);
            % title('I modified');
        end

         TTmax(:,nt+1) = T_Gauss(:,round(Ny/2),1)';

         x_L_ini = x_L_ini + V_L * Dt;
         T_ini = T_Gauss;
    end

    if ispc
        output0 = [linspace(-Lw,Le,Nx)', TTmax(:,:)];
        save('C:\Reza Karimi\Prof Toyserkani\LPB_MODELING\test11.dat', 'output0', '-ascii');
        TData = load ('C:\Reza Karimi\Prof Toyserkani\LPB_MODELING\test11.dat');

        x1 = TData(:,1);

        figure;
        for t = 1 : nt + 1
            TTT = TData(: , t+1);
            hold on;
            plot(x1 , TTT);
        end
        xlabel('x [m]');
        ylabel('T_{x, Laser} [C]');
        title('T-t distributions]');
        grid on; box on;
    end


    [xx, yy] = meshgrid(linspace(-Lw, Le, Nx), linspace(-Ls, Ln, Ny));

    figure,
    tmp_1 = [];
    tmp_2 = [];
    tmp_3 = [];
    i = 1;
    for x = unique(xx(1,:))
        j = 1;
        for y = unique(yy(:,:))'
            tmp_1(j,i) = fun_Q(x * scale_factor_alpha,y * scale_factor_beta);
            tmp_2(j,i) = fun_I(x,y);
            tmp_3(j,i) = fun_I_modified(x,y);
            j = j + 1;
        end
            i = i + 1;
    end


    ax_1 = subplot(2,2,1), surf(xx, yy, Q_Comb'), title('Q Comb');
    ax_2 = subplot(2,2,2), surf(xx, yy, tmp_1),  title('Q Fit');
    ax_3 = subplot(2,2,3), surf(xx, yy, tmp_2),  title('I');
    ax_4 = subplot(2,2,4), surf(xx, yy, tmp_3),  title('I modified');

    % linkaxes([ax_1, ax_2],'z');
    % linkaxes([ax_3, ax_4],'z');
    % linkaxes([ax_1, ax_2, ax_3, ax_4],'xy');






    figure
    subplot(2,2,1)
    surfc(xx,yy,Q_Conv');
    colorbar
    xlabel('x [m]');
    ylabel('y [m]');
    title('Convection [W/m^2]]');
    pbaspect([Length/Width 1 1]);

    subplot(2,2,2)
    surfc(xx,yy,Q_Rad');
    colorbar
    xlabel('x [m]');
    ylabel('y [m]');
    title('Radiation [W/m^2]');
    pbaspect([Length/Width 1 1]);

    subplot(2,2,3)
    surfc(xx,yy,Q_Melt');
    colorbar
    xlabel('x [m]');
    ylabel('y [m]');
    title('Melting [W/m^2]');
    pbaspect([Length/Width 1 1]);

    subplot(2,2,4)
    surfc(xx,yy,Q_Comb');
    colorbar
    xlabel('x [m]');
    ylabel('y [m]');
    title('Combined [W/m^2]');
    pbaspect([Length/Width 1 1]);

    figure,
    subplot(2,2,1);
    surfc(xx,yy,T_Ros(:,:,1)')
    colorbar;
    xlabel('x [m]');
    ylabel('y [m]');
    zlabel('T_{Ros1} [C]');
    title('T_{Ros1} @ layer 1');
    pbaspect([Length/Width 1 1]);

    subplot(2,2,2);
    surf(xx,yy,T_Gauss(:,:,1)');
    colorbar
    xlabel('x [m]');
    ylabel('y [m]');
    zlabel('T_{Gauss} [C]');
    title('T_{Gauss} @ layer 1');
    pbaspect([Length/Width 1 1])

    subplot(2,2,3)
    surfc(xx,yy,T_Ros(:,:,2)')
    colorbar
    xlabel('x');
    ylabel('y');
    zlabel('T_{Ros2} [C]');
    title('T_{Ros2} @ layer 2');
    pbaspect([Length/Width 1 1]);

    subplot(2,2,4)
    surfc(xx,yy,T_Gauss(:,:,2)')
    colorbar
    xlabel('x [m]');
    ylabel('y [m]');
    zlabel('T_{Gauss} [C]');
    title('T_{Gauss} @ layer 2');
    pbaspect([Length/Width 1 1])




    % SURFACE TENSION
    Sig_m = 2;
    Sig_Slope = 3.76e-4;

    for j = 1 : Ny
        for i = 1: Nx
            if T_Gauss(i,j,1) > Tmelt,
                Sig(i,j) = 2 + Sig_m - Sig_Slope * (T_Gauss(i,j,1) - Tmelt);
            else;
                Sig(i,j) = 2;
            end
        end
    end

%   figure
%   level_num = 20;
%   %contour (xx,yy,Sig,level_num )
%   hold on
%   %surf (xx,yy,Sig);
%
%   h = .1;
%   [ U, V ] = gradient (Sig,h);
%   quiver (xx,yy,U,V);
%
%   view ( 3 );
%   hold off
%
%    %keyboard
%
%   xlabel ( 'x [m]' );
%   ylabel ( 'y [m]' );
%   zlabel ( '\sigma [N/m]' );
%   title ( '\sigma Contours & gradient Vectors' )
%
%     %----

    [xx,yy]=meshgrid(linspace(-Lw,Le,Nx),linspace(0,Ln,round(Ny/2)));

    for j=1:round(Ny/2)
        jj=j-1+round(Ny/2);
        for i=1:Nx
            TT5(i,j)=T_Gauss(i,jj,1);
        end
    end

    Tmin = round(min(T_Gauss(:)));
    Tmax = round(max(T_Gauss(:)));
    Tlevs = [Tini_scalar 100 200 500 1000 2000 2500 3000 3500 5000 Tmax];
    Tindex = [Tmelt - DTm/2 Tmelt+DTm/2];

    figure
    subplot(2,1,1)
    [C,h]=contourf(xx,yy,TT5', Tlevs);
    clabel(C,h, 'edgecolor','none');
    colorbar;
    xlabel('x [m]');
    ylabel('y [m]');
    title('T_{Gausee} @ layer 1');
    hold on

    contourc(TT5',[1400 1400])
    [C,h]=contour(xx,yy,TT5',Tindex,'LineWidth',2,'LineColor', 'Red');
    clabel(C,h)
    hold off

    [xx,zz]=meshgrid(linspace(-Lw,Le,Nx),linspace(0,-Height,Nz));


    for k=1:Nz
        for i=1:Nx
            TT6(i,k)=T_Gauss(i,round(Ny/2),k);
        end
    end

    subplot(2,1,2)
    [C,h]=contourf(xx,zz,TT6', Tlevs);
    clabel(C,h, 'edgecolor','none');
    colorbar;
    xlabel('x [m]');
    ylabel('z [m]');
    title('T_{Gausee} @ layer 1');
    hold on
    [C,h]=contour(xx,zz,TT6',Tindex,'LineWidth',2,'LineColor','Red');
    clabel(C,h)
    hold off

    %figure; contour(xx,yy,Q_Conv',10);hold on;h=1e-1;[U,V]=gradient(Q_Conv',h);quiver(xx,yy,U,V);view(3);hold off
    %keyboard




% --------------------------------------------------------------------
function params = fit2DGaussianToQ(Q, initial_guess_magnitude)
% --------------------------------------------------------------------
    S = Q;
    S_bu = S;
    keyboard
    % S_bu = S;
    % S = zeros(11,11);
    % S(6,6) = 10e2;
    % S(end-4:end,:) = S(1:5,:);
    n = size(S,2)-1;
    m = size(S,1)-1;
    % A0 = [initial_guess_magnitude,0,50,0,50,0]; % Inital (guess) parameters
    A0 = [initial_guess_magnitude,0,1,0,1,0]; % Inital (guess) parameters
    % A0 = [10e2,0,50,0,50,0]; % Inital (guess) parameters

    lb = [0,-n/2,0,-n/2,0,0];
    ub = [realmax('double'),n/2,(n/2)^2,n/2,(n/2)^2,pi/4];

    [x,y]=meshgrid(-n/2:n/2,-m/2:m/2);
    X=zeros(m+1,n+1,2);
    X(:,:,1)=x;
    X(:,:,2)=y;

    g = @(A,X) A(1)*exp( -((X(:,:,1)-A(2)).^2/(2*A(3)^2) + (X(:,:,2)-A(4)).^2/(2*A(5)^2)) );
    f = @(A,X) A(1)*exp( -(...
        ( X(:,:,1)*cos(A(6))-X(:,:,2)*sin(A(6)) - A(2)*cos(A(6))+A(4)*sin(A(6)) ).^2/(2*A(3)^2) + ...
        ( X(:,:,1)*sin(A(6))+X(:,:,2)*cos(A(6)) - A(2)*sin(A(6))-A(4)*cos(A(6)) ).^2/(2*A(5)^2) ) );
    % [A,resnorm,res,flag,output] = lsqcurvefit(g,A0(1:5),X,S',lb(1:5),ub(1:5));
    % [A,resnorm,res,flag,output] = lsqcurvefit(g,A0(1:5),X,S');
    % keyboard
    options = optimoptions('lsqcurvefit','MaxFunctionEvaluations',10000);
    % [A,resnorm,res,flag,output] = lsqcurvefit(f,A0,X,S,lb,ub,options);
    [A,resnorm,res,flag,output] = lsqcurvefit(g,A0,X,S,lb,ub,options);
    params = A;

    % Q_recon = f(params, X);
    Q_recon = g(params, X);

    figure,
    subplot(1,2,1), surf(x,y,S), title('Q Comb')
    subplot(1,2,2), surf(x,y,Q_recon), title('Q Recon')
    suptitle('Fitting Process')















