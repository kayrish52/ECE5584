figure(1)
bar(hBinsCenter,hCount{5,10})
xticks(hBinsCenter)
xticklabels({'0.0625','0.1875','0.3125','0.4375','0.5625','0.6875',...
    '0.8125','0.9375'})
xlabel('Histogram Bin Centers')
ylabel('Counts per Bin')
title('H Values')

figure(2)
bar(sBinsCenter,sCount{5,10})
xticks(sBinsCenter)
xticklabels({'0.125','0.3875','0.625','0.875'})
xlabel('Histogram Bin Centers')
ylabel('Counts per Bin')
title('S Values')

figure(3)
bar(vBinsCenter,vCount{5,10})
xticks(vBinsCenter)
xticklabels({'0.25','0.75'})
xlabel('Histogram Bin Centers')
ylabel('Counts per Bin')
title('V Values')

figure(4)
image(ims{5,10})
title(sprintf('Class: %s','Deer'))

