% -------------------------------------------------------------------------
function LPB6()
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

    Dt = 1;

    % Ambient Conditions
    Tini_scalar=20;                                % Initial temperature [C]
    Tamb_scalar=20;                                % Ambient temperature [C]
    hc=20;                                  % Convective heat transfer coefficint [W/m^2.C]
    Sigma=5.669e-8;                         % Estefan-Boltzmann Const. W/m^2.K^4

    % LASER
    P_L=200;                                % Laser power [W]
    R_L=1e-3;                               % Laser spot diamter (on the powder surface) [m^2]
    V_L=1e-3;                               % Laser scanning velocity [m/s]

    % MELT
    Beta=0.35;                              % Surface absrbtivity
    % Tmelt=1400;                             % Powder melting point [C]
    Tmelt=140;                             % Powder melting point [C]
    Tboil=7000;                             % Boiling temperature [C]
    Lambda_SL=204500;                       % Powder latent heat of fusion [J/kg
    Lambda_LV=Lambda_SL;                    % CHANGE THIS LATER


    % DIMENSIONS (Laser beam position initially at the intesections of Lw, Le, Ln, Ls and at z=0)
    Lw=4*R_L;
    Le=6*R_L;
    Ln=4*R_L;
    Ls=4*R_L;

    Length=Le+Lw;
    Width=Ln+Ls;
    Height=1e-3;

    % BED PROPERTIES
    Rho_P=8440;                             % powder density [kg/m^3]
    Rho_A=1;                                % Inert gas density [kg/m^3]
    Cp_P=410;                               % Powder specific heat [J/kg.C]
    Cp_A=1000;                              % Inert gas specific heat [J/kg.C]
    k_P=10;                                 % Powder thermal conductivity [W/m.C]
    k_A=0.02;                               % Inert gas thermal conductivity [W/m.C]
    Phi=0.5;                                % Powder porosoity [%]

    % BULK (POWDER + INERT GAS) PROPERTIES
    Rho_B=Rho_P*(1-Phi)+Rho_A*Phi;          % Bulk density [kg/m^3]
    Cp_B=Cp_P*(1-Phi)+Cp_A*Phi;             % Bulk specific [J/kg.C]
    k_B=k_P*(1-Phi)+k_A*Phi;                % Bulk thermal conductivity [W/m.C]
    Alfa_B=k_B/(Rho_B*Cp_B);                % Bulk thermal diffusivity [m^2/s]

    % No of increments in each direction
    nx = 20; % 20;
    ny = 20; % 14;
    nz = 3; % 10;

    Dx=Length/nx;
    Dy=Width/ny;
    Dz=Height/nz;

    % Total No of nodes in the domain
    Nx=nx+1;
    Ny=ny+1;
    Nz=nz+1;

    A_L=pi*R_L^2;                           % Laser spot area [m^2]


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

    for t = 1:2,
        counter = 1;
        total_count = Nz * Ny * Nx;
        Q_Conv(:,:) = hc * (T_Gauss(:,:,1) - T_amb(:,:,1));
        Q_Rad(:,:) = Sigma * Beta * ((T_Gauss(:,:,1) + 273.15).^4 - (T_amb(:,:,1)+273.15).^4);
        Q_melt = sum(T_Gauss > Tmelt, 3) * Lambda_SL * Rho_B * Dz;
        Q_Comb(:,:) = Q_Conv + Q_Rad + Q_melt;
        params = fit2DGaussianToQ(Q_Comb, 2 * P_L * Beta / (pi * R_L^2))
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

    %                 if R_Ros < 1e-9,
    %                     R_Ros=Dz/10;
    %                 end;

                    % ------------------------------------------------------------------
                    % Energy flux / distribution delivered to bed. (Watt / m^2)
                    % ------------------------------------------------------------------
                    II(i,j) = 2 * P_L * Beta / (pi * R_L^2) * exp(-2 * (Kisi^2 + y^2) / R_L^2);

                    % ------------------------------------------------------------------
                    % T Rosenthal
                    % ------------------------------------------------------------------
                    T_Ros(i,j,k) = T_ini(i,j,k) + P_L * Beta / (2 * pi * k_B * R_Ros) * exp(- V_L * (Kisi + R_Ros) / (2 * Alfa_B));

    %                 if counter == 116
    %                     T_Ros
    %                     keyboard
    %                 end
                    % ------------------------------------------------------------------
                    % T Gauss
                    % ------------------------------------------------------------------


                    % num=19;
                    % TTTT = Int_2D(num,Le,Lw,Ls,Ln,V_L,R_L,R_Ros,Kisi,Eta,Zeta, Alfa_B)

                    % TXY=TTTT
                    % keyboard


                    % constant = P_L * Beta / (pi^2 * k_B * R_L^2);
                    % fun_R = @(alpha_, beta_) sqrt( (x - (x_L_fin + alpha_)).^2 + (y - beta_).^2 + z.^2);
                    % fun_kolli = @(alpha_, beta_) 1 ./ fun_R(alpha_, beta_) .* exp(-2 .* (alpha_.^2 + beta_.^2) / R_L.^2) .* exp(- V_L .* (x - (x_L_fin + alpha_) + fun_R(alpha_, beta_)) / (2*Alfa_B));

                    constant = 1;
                    fun_R = @(alpha_, beta_) sqrt( (x - (x_L_fin + alpha_)).^2 + (y - beta_).^2 + z.^2);
                    % fun_Q = @(alpha_, beta_) 10e6;
                    fun_Q = @(alpha_, beta_) params(1) * exp( -((alpha_ - params(2)) .^ 2 / (2 * params(3) ^ 2) + (beta_ - params(4)) .^ 2 / (2 * params(5) ^ 2)) );
                    fun_I_modified = @(alpha_, beta_) (2 * P_L * Beta / (pi * R_L^2) .* exp(-2 .* (alpha_.^2 + beta_.^2) / R_L.^2) - fun_Q(alpha_, beta_));
                    fun_kolli = @(alpha_, beta_) ...
                        (2 * P_L * Beta / (pi * R_L^2) .* exp(-2 .* (alpha_.^2 + beta_.^2) / R_L.^2) - fun_Q(alpha_, beta_)) .* ...
                        (1 / (2 * pi * k_B)) ./ fun_R(alpha_, beta_) .* exp(- V_L .* (x - (x_L_fin + alpha_) + fun_R(alpha_, beta_)) / (2*Alfa_B));


                    alpha_min = - R_L;
                    alpha_max = R_L;
                    beta_min = @(alpha_) - sqrt(R_L.^2 - alpha_.^2);
                    beta_max = @(alpha_) + sqrt(R_L.^2 - alpha_.^2);
    %                 alpha_min = - Lw;
    %                 alpha_max = Le;
    %                 beta_min = -Ls;
    %                 beta_max = Ln;

                    T_Gauss(i,j,k) = T_ini(i,j,k) + constant * integral2(fun_kolli,alpha_min,alpha_max,beta_min,beta_max);

                    % ------------------------------------------------------------------
                    % Others....
                    % ------------------------------------------------------------------
    %                 if T_Gauss(i,j,k) > Tmelt,
    %                     Qsens = Rho_B * Dx * Dy * Dz * Cp_B * (T_Gauss(i,j,k) - Tmelt);
    %                     Qlat = Rho_B * Dx * Dy * Dz * Lambda_SL;
    %                     T_Gauss(i,j,k) = Tmelt + (Qsens - Qlat) /(Rho_B * Dx * Dy * Dz * Cp_B);
        % %                 if T_Gauss(i,j,k) > Tboil,
        % %                     Qsens = Rho_B * Dx * Dy * Dz * Cp_B * (T_Gauss(i,j,k) - Tboil);
        % %                     Qlat = Rho_B * Dx * Dy * Dz * Lambda_LV;
        % %                     T_Gauss(i,j,k) = Tboil + (Qsens - Qlat) /(Rho_B * Dx * Dy * Dz * Cp_B);
    %     % %                 end;
    %                 end;

                    Sum_E1 = Sum_E1 + Rho_B * Cp_B * Dx * Dy * Dz * (T_Ros(i,j,k) - Tini_scalar) * Dt;
                    Sum_E2 = Sum_E2 + Rho_B * Cp_B * Dx * Dy * Dz * (T_Gauss(i,j,k) - Tini_scalar) * Dt;

                    fprintf('Iteration #%03d/%03d \t \t %.3f \t %.3f \t \t %.3f \t \t %.3f\n', counter, total_count, t, Sum_E1, Sum_E2, Sum_E1/Sum_E2);
                    % [Sum_E1 Sum_E2 Sum_E1/Sum_E2];
                    counter = counter + 1;
                end
            end
        end
        % T = T_Gauss;
        % x_L_ini = x_L_fin;
        [xx,yy]=meshgrid(linspace(-Lw,Le,Nx),linspace(-Ls,Ln,Ny));
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

        figure,
        tmp = [];
        i = 1;
        for x = unique(xx(1,:))
            j = 1;
            for y = unique(yy(:,:))'
                tmp(j,i) = fun_I_modified(x,y);
                j = j + 1;
            end
            i = i + 1;
        end
        surf(xx,yy,tmp);
        title('I modified');




    end

    % [xx,yy]=meshgrid(linspace(-Lw,Le,Nx),linspace(-Ls,Ln,Ny));
    % xlabel('x [m]');
    % ylabel('y [m]');

    % % TT1=T_Ros(:,:,1);
    % % TT2=T_Ros(:,:,2);
    % % TT3=T_ini(:,:,1); % T_Gauss(:,:,1);
    % % TT4=T_ini(:,:,2); % T_Gauss(:,:,2);

    % % figure,
    % % surfc(xx,yy,II');
    % % colorbar;
    % % xlabel('x [m]');
    % % ylabel('y [m]');
    % % zlabel('I [W/m^2]');
    % % title('Laser Power Distribution [W/m^2]');
    % % pbaspect([Length/Width 1 1]);

    % figure,
    % subplot(2,2,1);
    % surfc(xx,yy,T_Ros(:,:,1)')
    % colorbar;
    % xlabel('x [m]');
    % ylabel('y [m]');
    % zlabel('T_{Ros1} [C]');
    % title('T_{Ros1} @ layer 1');
    % pbaspect([Length/Width 1 1]);

    % subplot(2,2,2);
    % surf(xx,yy,T_Gauss(:,:,1)');
    % colorbar
    % xlabel('x [m]');
    % ylabel('y [m]');
    % zlabel('T_{Gauss} [C]');
    % title('T_{Gauss} @ layer 1');
    % pbaspect([Length/Width 1 1])

    % subplot(2,2,3)
    % surfc(xx,yy,T_Ros(:,:,2)')
    % colorbar
    % xlabel('x');
    % ylabel('y');
    % zlabel('T_{Ros2} [C]');
    % title('T_{Ros2} @ layer 2');
    % pbaspect([Length/Width 1 1]);

    % subplot(2,2,4)
    % surfc(xx,yy,T_Gauss(:,:,2)')
    % colorbar
    % xlabel('x [m]');
    % ylabel('y [m]');
    % zlabel('T_{Gauss} [C]');
    % title('T_{Gauss} @ layer 2');
    % pbaspect([Length/Width 1 1])

    % % figure
    % % subplot(3,1,1)
    % % [C,h]=contourf(xx,yy,II');
    % % clabel(C,h)
    % % colorbar
    % % xlabel('x [m]');
    % % ylabel('y [m]');
    % % title('Laser Power Density [W/m^2]');

    % % subplot(3,1,2)
    % % [C,h]=contourf(xx,yy,Q_Conv');
    % % clabel(C,h)
    % % colorbar
    % % xlabel('x [m]');
    % % ylabel('y [m]');
    % % title('Convection [W]]');

    % % subplot(3,1,3)
    % % [C,h]=contourf(xx,yy,Q_Rad');
    % % clabel(C,h)
    % % colorbar
    % % xlabel('x [m]');
    % % ylabel('y [m]');
    % % title('Radiation [W]');

    % % [xx,yy]=meshgrid(linspace(-Lw,Le,Nx),linspace(0,Ln,round(Ny/2)));

    % % for j=1:round(Ny/2),
    % %     jj=j-1+round(Ny/2);
    % %     for i=1:Nx,
    % %         TT5(i,j)=T_Gauss(i,jj,1);
    % %     end;
    % % end;

    % % Tmin = min(T_Gauss(:));
    % % Tmax = max(T_Gauss(:));
    % % Tinc = (Tmax - Tmin) / 10;
    % % Tlevs = Tmin:Tinc:Tmax;
    % % Tindex = Tmelt:(Tboil-Tmelt):Tboil;

    % % figure
    % % subplot(2,1,1)
    % % [C,h]=contourf(xx,yy,TT5', Tlevs);
    % % clabel(C,h, 'edgecolor','none');
    % % colorbar;
    % % xlabel('x [m]');
    % % ylabel('y [m]');
    % % title('T_{Gausee} @ layer 1');
    % % hold on
    % % [C,h]=contour(xx,yy,TT5',Tindex,'LineWidth',2,'LineColor', 'Red');
    % % clabel(C,h)
    % % hold off


    % % [xx,zz]=meshgrid(linspace(-Lw,Le,Nx),linspace(0,-Height,Nz));


    % % for k=1:Nz,
    % %     for i=1:Nx,
    % %         TT6(i,k)=T_Gauss(i,round(Ny/2),k);
    % %     end;
    % % end;

    % % subplot(2,1,2)
    % % [C,h]=contourf(xx,zz,TT6', Tlevs);
    % % clabel(C,h, 'edgecolor','none');
    % % colorbar;
    % % xlabel('x [m]');
    % % ylabel('z [m]');
    % % title('T_{Gausee} @ layer 1');
    % % hold on
    % % [C,h]=contour(xx,zz,TT6',Tindex,'LineWidth',2,'LineColor','Red');
    % % clabel(C,h)
    % % hold off


    % % Sum=0;
    % % for j=1:Ny,
    % %     for i=1:Nx,
    % %         Sum=Sum+II(i,j)*Dx*Dy * t_Process;
    % %     end;
    % % end;
    % % Sum

    % % close figure 1


% --------------------------------------------------------------------
function params = fit2DGaussianToQ(Q, initial_guess_magnitude)
% --------------------------------------------------------------------
    S = Q;
    n = size(S,1)-1;
    m = size(S,2)-1;
    A0 = [initial_guess_magnitude,0,50,0,50,0]; % Inital (guess) parameters

    lb = [0,-n/2,0,-n/2,0,0];
    ub = [realmax('double'),n/2,(n/2)^2,n/2,(n/2)^2,pi/4];

    [x,y]=meshgrid(-n/2:n/2,-m/2:m/2);
    X=zeros(m+1,n+1,2);
    X(:,:,1)=x;
    X(:,:,2)=y;

    g = @(A,X) A(1)*exp( -((X(:,:,1)-A(2)).^2/(2*A(3)^2) + (X(:,:,2)-A(4)).^2/(2*A(5)^2)) );
    [A,resnorm,res,flag,output] = lsqcurvefit(g,A0(1:5),X,S,lb(1:5),ub(1:5));
    params = A;



















