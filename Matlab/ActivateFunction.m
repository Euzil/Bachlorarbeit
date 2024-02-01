figure('NumberTitle', 'off', 'Name', 'Sigmoid');
x=-10:0.1:10;
y= 1 ./ (1 + exp(-x));
plot(x,y);
xlabel('X');ylabel('Y');%坐标轴表示对象标签
grid on;%显示网格线
axis on;%显示坐标轴
axis([-8,8,0,1]);%x,y的范围限制
title('Sigmoid');

figure('NumberTitle', 'off', 'Name', 'Tanh');
x=-5:0.1:5;
y=2./(1+exp(-2*x))-1;
plot(x,y);
xlabel('X');ylabel('Y');%坐标轴表示对象标签
grid on;%显示网格线
axis on;%显示坐标轴
axis([-5,5,-1,1]);%x,y的范围限制
title('Tanh');

figure('NumberTitle', 'off', 'Name', 'ReLU');
x=-5:0.1:5;
y=max(0,x);
plot(x,y);
xlabel('X');ylabel('Y');%坐标轴表示对象标签
grid on;%显示网格线
axis on;%显示坐标轴
axis([-5,5,-5,5]);%x,y的范围限制
title('ReLU');

figure('NumberTitle', 'off', 'Name', 'Threshold');
x1=-5:0.1:0;
x2=0:0.1:5;
y1=0*x1;
y2=1+(x2-x2);
x=[x1 x2];
y=[y1 y2];
plot(x,y);
xlabel('X');ylabel('Y');%坐标轴表示对象标签
grid on;%显示网格线
axis on;%显示坐标轴
axis([-5,5,-1,1]);%x,y的范围限制
title('Threshold');