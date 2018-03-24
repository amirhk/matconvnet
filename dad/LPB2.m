% Laser Powder Bed Additive Manufacturing
clear all;
close all;

%DESIGN & OPERATING CONDITIONS

% Ambient Conditions
Tini=20;                                % Initial temperature [C]
Tamb=20;                                % Ambient temperature [C]
hc=20;                                  % Convective heat transfer coefficint [W/m^2.C]
Sigma=5.669e-8;                         % Estefan-Boltzmann Const. W/m^2.K^4

% LASER
P_L=1000;                               % Laser power [W]
R_L=0.5e-3;                             % Laser spot diamter (on the powder surface) [m^2]
V_L=1e-3;                               % Laser scanning velocity [m/s]

% MELT
Beta=0.35;                              % Surface absrbtivity
Tmelt=1400;                             % Powder melting point [C]
Lambda=204500;                          % Powder latent heat of fusion [J/kg]

% DIMENSIONS (Laser beam position initially at the intesections of Lw, Le, Ln, Ls and at z=0)
Lw=4*R_L*2;
Le=4*R_L*2;
Ln=2*R_L*2;
Ls=2*R_L*2;

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
nx=10*2; % 35;
ny=10*2; % 20;
nz=5; % 10;

Dx=Length/nx;
Dy=Width/ny;
Dz=Height/nz;

% Total No of nodes in the domain
Nx=nx+1;
Ny=ny+1;
Nz=nz+1;

A_L=pi*R_L^2;                           % Laser spot area [m^2]


% INITIALIZATION (TEMPERATURES, HEAT SOURCES, AND STATE OF PHASE)
% NOTE: 2 extra/fictitious nodes are considered in each direction (x,y, and z)
%
% for k=1:Nz+1,
%     for j=1:Ny+1,
%         for i=1:Nx+1,
%             T(i,j,k)=Tini;
%             Tnew(i,j,k)=T(i,j,k);
%         end;
%     end;
% end;

% T = Tini * ones(Nx, Ny, Nz);
% Tnew = T;


% Just ONE Iteration for now
% N_Itr=1;
Dt=1;     % the laser is still within the doamin


Sum_E1=0;   % Stored energy - Rosenthal's equation
Sum_E2=0;   % Stored energy - Gaussian
Sum_E3=0;   % Heat dissipation - Convection
Sum_E4=0;   % Heat dissipation - Radiation

x_i_laser = 0;
y_i_laser = 0;
z_i_laser = 0;

x_f_laser = x_i_laser + V_L * Dt;
y_f_laser = 0;
z_f_laser = 0;
% keyboard

% SIMULATION STARTS HERE
% for kk=1:N_Itr,
counter = 1;
total_count = Nz * Ny * Nx;
for k=1:Nz,
    for j=1:Ny,
        for i=1:Nx,

            % Iterating over all points in the space
            x = (i-1) * Dx - Lw;
            y = (j-1) * Dy - Ls;
            z = (k-1) * Dz;

            % This represents the final point of the laser
            Kisi = x - x_f_laser;
            Eta = y - y_f_laser;
            Zeta = z - z_f_laser;

            R_Ros = sqrt(Kisi^2 + Eta^2 + Zeta^2);
            % R_Ros=sqrt((x - x_i_laser)^2+(y - y_i_laser)^2+(z - z_i_laser)^2);
            % R=sqrt((x1 - Kisi)^2 + (y1 - Eta)^2 + z1^2);

            if R_Ros==0,
                R_Ros=Dz/10;
            end;

            % Energy flux / distribution delivered to bed. (Watt / m^2)
            II(i,j)=2 * P_L * Beta / (pi * R_L^2) * exp(-2 * (x^2 + y^2) / R_L^2);

            % CHECK the follwing Equations carefully.
            T_Ros(i,j,k) = Tini + P_L * Beta / (2 * pi * k_B * R_Ros) * exp(- V_L * (Kisi + R_Ros) / (2 * Alfa_B));


            % I think R in Ros. eq. should take 3 dimensions into account
            % e.g. RR=RR and NOT sqrt(xx^2+yy^2)

            % fun=@(x,y) exp(-2.*(x.^2+y.^2) / R_L.^2) * 1/sqrt((x-Kisi).^2+y.^2+Zeta^2) * exp(-V_L.*(x-Kisi+R)/(2*Alfa_B));
            % fun=@(x,y) exp(-2.*(x.^2+y.^2) / R_L.^2) * 1/sqrt((x-Kisi).^2+y.^2+Zeta^2) * exp(-V_L.*(x-Kisi+sqrt((x-Kisi).^2+y.^2+Zeta^2))/(2*Alfa_B));


            constant = P_L / (pi^2 * R_L^2 * k_B);
            fun_R = @(alpha_, beta_) sqrt( (x - (x_f_laser + alpha_)).^2 + (y - beta_).^2 + z.^2);
            fun_kolli = @(alpha_, beta_) 1 ./ fun_R(alpha_, beta_) .* exp(-2 .* (alpha_.^2 + beta_.^2) / R_L.^2) .* exp(- V_L .* (x - (x_f_laser + alpha_) + fun_R(alpha_, beta_)) / (2*Alfa_B));


            alpha_min = - R_L;
            alpha_max = R_L;
            beta_min = @(alpha_) - sqrt(R_L.^2 - alpha_.^2);
            beta_max = @(alpha_) sqrt(R_L.^2 - alpha_.^2);

            % alpha_min = @(beta_) - sqrt(R_L.^2 - beta_.^2);
            % alpha_max = @(beta_) sqrt(R_L.^2 - beta_.^2);
            % beta_min = - R_L;
            % beta_max = R_L;

            H = constant * integral2(fun_kolli,alpha_min,alpha_max,beta_min,beta_max,'Method','iterated','AbsTol',0,'RelTol',1e-2);
            T2(i,j,k)=Tini +1/(2*pi*k_B)*H;

            Sum_E1=Sum_E1+Rho_B*Cp_B*Dx*Dy*Dz*(T_Ros(i,j,k)-Tini)/Dt;
            Sum_E2=Sum_E2+Rho_B*Cp_B*Dx*Dy*Dz*(T2(i,j,k)-Tini)/Dt;

            fprintf('Iteration #%03d/%03d \t %.3f \t %.3f \t %.3f\n', counter, total_count, Sum_E1, Sum_E2, Sum_E1/Sum_E2);
            % [Sum_E1 Sum_E2 Sum_E1/Sum_E2];
            counter = counter + 1;
        end;
    end;
end;
% end;

% [xx,yy]=meshgrid(x1,y1);
% [xx,yy]=meshgrid(1:Nx,1:Ny);
[xx,yy]=meshgrid(linspace(-Lw,Le,Nx),linspace(-Ls,Ln,Ny));
xlabel('x [mm]');
ylabel('y [mm]');

for j=1:Ny,
    for i=1:Nx,
        TT1(i,j)=T_Ros(i,j,1);
        TT2(i,j)=T_Ros(i,j,2);
        TT3(i,j)=T2(i,j,1);
    end;
end;

figure,

subplot(2,3,1)
surfc(xx,yy,II')
colorbar
xlabel('x');
ylabel('y');
zlabel('I [W/m^2]');
pbaspect([1 1 1]);

subplot(2,3,2)
surfc(xx,yy,T_Ros(:,:,1)')
colorbar
xlabel('x');
ylabel('y');
zlabel('T_{Ros1} [C]');
title('T_{Ros1} @ layer 1');
pbaspect([1 1 1]);

subplot(2,3,3)
surfc(xx,yy,T_Ros(:,:,2)')
colorbar
xlabel('x');
ylabel('y');
zlabel('T_{Ros2} [C]');
title('T_{Ros2} @ layer 2');
pbaspect([1 1 1]);

subplot(2,3,5)
surfc(xx,yy,T2(:,:,1)')
colorbar
xlabel('x');
ylabel('y');
zlabel('T_{Gauss} [C]');
title('T_{Gauss} @ layer 1');
pbaspect([1 1 1])

subplot(2,3,6)
surfc(xx,yy,T2(:,:,2)')
colorbar
xlabel('x');
ylabel('y');
zlabel('T_{Gauss} [C]');
title('T_{Gauss} @ layer 2');
pbaspect([1 1 1])

for j=1:Ny,
    for i=1:Nx,
        Sum_E3=Sum_E3+hc*Dx*Dy*(T2(i,j,1)-Tamb)*Dt;
        Sum_E4=Sum_E4+Beta*Sigma*Dx*Dy*((T2(i,j,1)+273.15)^4-(Tini+273.15)^4)*Dt;
    end;
end;
[Sum_E3 Sum_E4 Sum_E3/Sum_E4]

Sum=0;
for j=1:Ny,
    for i=1:Nx,
        Sum=Sum+II(i,j)*Dx*Dy;
    end;
end;
Sum

