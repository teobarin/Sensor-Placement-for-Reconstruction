clear; close all; clc
datpath = '../DATA/';

load([datpath,'YaleB_32x32.mat']);
[nSmp,nFea] = size(fea);
% for i = 1:nSmp
%      fea(i,:) = fea(i,:) ./ max(1e-12,norm(fea(i,:)));
% end
%===========================================
maxValue = max(max(fea));
fea = fea/maxValue;
%===========================================
X=fea';