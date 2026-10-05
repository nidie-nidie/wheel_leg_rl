%%%%%%%%%%%%%%%%%%%%%%% 定义固定量 %%%%%%%%%%%%%%%%%%%%%%%%%

%参考哈工程开源（https://zhuanlan.zhihu.com/p/563048952）平面五连杆模型及其参数
l_1=0.21;           %单位：m
l_2=0.25;          %单位：m
l_3=l_2;            %单位：m
l_4=l_1;            %单位：m
l_AE=0.0;         %电机中心距,单位：m

%%%%%%%%%%%%%%%%%%%%%%% 定义变量 %%%%%%%%%%%%%%%%%%%%%%%%%%

syms phi1 phi2 phi3 phi4;   %求虚拟腿长和腿长角角度过程量
syms d_phi1 d_phi4;         %关节电机的角度
syms xc yc xd yd xb yb;     %求虚拟腿长和腿长角坐标过程量
syms F Tp;                  %求虚拟腿长受到的力和扭矩

%%%%%%%%%%% VMC 通过关节电机角度求虚拟腿长和腿长角 %%%%%%%%%%%

%描述变量之间的关系
xb=l_1*cos(phi1);
yb=l_1*sin(phi1);
xd=l_AE+l_4*cos(phi4);
yd=l_4*sin(phi4);

A0=2*l_2*(xd-xb);
B0=2*l_2*(yd-yb);
C0=l_2^2+(xd-xb)^2+(yd-yb)^2-l_3^2;
phi2=2*atan((B0+sqrt(A0^2+B0^2-C0^2))/(A0+C0));

xc=xb+l_2*cos(phi2);
yc=yb+l_2*sin(phi2);
l_0=sqrt((xc-l_AE/2)^2+yc^2);
phi0=atan2(yc,xc-l_AE/2);

%五连杆末端的位姿与关节电机的关系
position=[l_0,phi0];
matlabFunction(position,'File','leg_position');

%表示l_0和phi0与phi1和phi4之间的关系
J11=diff(l_0,phi1);
J12=diff(l_0,phi4);
J21=diff(phi0,phi1);
J22=diff(phi0,phi4);
JacobianMatrix=[J11 J12;J21 J22];

%五连杆末端位姿变化率与关节电机的关系
speed=JacobianMatrix*[d_phi1;d_phi4];
matlabFunction(speed,'File','leg_speed');

%五连杆末端受力与关节电机的关系
T=JacobianMatrix'*[F;Tp];
matlabFunction(T,'File','leg_force');