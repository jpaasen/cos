
load '/home/me/Work/UiO/Phd/Code/data/sonar/real/hugin/hisas1030/run080827_1-l04-s0b2p1300.mat'

fc = Sonar.fc;
c = Sonar.c;
x_r = Sonar.rx_x(:,2);

image = tdibMatlab( mfdata, mtaxe, x_r, D, c, fc, xarr, yarr );

imagesc(20*log10(abs(image)));