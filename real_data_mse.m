%%
  define_constants
%% 对于受试者的分析
load('Processed_data/RealData')
load('HRF.mat')
fNIRS_data = NirsClass('C:\Users\yp181\Desktop\demo\motion_artifacts_1');
fNIRS_data  =   Downsample(fNIRS_data);
SD          =   fNIRS_data.SD;
    t           =   fNIRS_data.t;
    fs          =   1 / (t(2) - t(1));
    fprintf('fs is %f\n',fs);
    tIncMan     =   ones(size(t));
    s           =   zeros(1,length(t));
    s((rt+1):512:length(t)-512) = 1;
  
  %%
[dc_no_crct, ~, ~, ~, ~, ~] = hmrBlockAvg(net_input.dc, s', t, [-39/fs_new (512-40)/fs_new] );
[dc_act_no_crct,~ , tHRF, ~, ~, ~] = hmrBlockAvg(net_input.dc_act, s', t, [-39/fs_new (512-40)/fs_new] );
  %% 其他的方法比较
  %%cbsi
  [dc_avg_cbsi,n_MA]         =   proc_Cbsi(net_input.dc_act,s,SD,t,tIncMan,OD_thred,STD);
  %%pca99
  [dc_avg_pca99,n_MA] =  proc_PCA(net_input.dc_act, s, SD, t, tIncMan,STD, OD_thred, 0.99);
  %%pca97
 [dc_avg_pca97,n_MA]  =   proc_PCA(net_input.dc_act, s, SD, t, tIncMan,STD, OD_thred, 0.97);
  %%spline
  [dc_avg_spline,n_MA]    =   proc_Spline(net_input.dc_act, s, SD, t, tIncMan, STD,OD_thred,0.99);
  %%wavlet
  [dc_avg_wavlet,n_MA] =   proc_Wavelet(net_input.dc_act,s,SD,t,tIncMan,STD,OD_thred,0.75);
  %%rloess
  [dc_avg_rloess,n_MA] = proc_rloess(net_input.dc_act, s, SD, t, tIncMan, STD, OD_thred);
  %% mse
 HbO_mse_no_crct     =   [];
HbO_mse_Spline      =   [];
HbO_mse_Wavelet01   =   [];
HbO_mse_PCA99      =   [];
HbO_mse_PCA97       =   [];
HbO_mse_Cbsi        =   [];
HbO_mse_DAE          =   [];
HbO_mse_rloess         =   [];
HbO_mse_LSTM_AE          =   [];

HbR_mse_no_crct     =   [];
HbR_mse_Spline      =   [];
HbR_mse_Wavelet01   =   [];
HbR_mse_PCA99      =   [];
HbR_mse_PCA97       =   [];
HbR_mse_Cbsi        =   [];
HbR_mse_LSTM_AE          =   [];
HbR_mse_rloess         =   [];
HbR_mse_DAE          =   [];
%% HRF

HbO_real        =   repmat(HRFs.HbO(1:512),14,1);
 HbR_real        =   repmat(HRFs.HbR(1:512),14,1);
 
 
HbO_no_crct     =   squeeze(dc_act_no_crct(1:512,1,:))';
HbO_Spline      =   squeeze(dc_avg_spline(1:512,1,:))';
HbO_Wavelet01   =   squeeze(dc_avg_wavlet(1:512,1,:))';
HbO_PCA99      =   squeeze(dc_avg_pca99(1:512,1,:))';
HbO_PCA97       =   squeeze(dc_avg_pca97(1:512,1,:))';
HbO_Cbsi        =   squeeze(dc_avg_cbsi(1:512,1,:))';
HbO_rloess        =   squeeze(dc_avg_rloess(1:512,1,:))';

HbR_no_crct     =   squeeze(dc_act_no_crct(1:512,2,:))';
HbR_Spline      =   squeeze(dc_avg_spline(1:512,2,:))';
HbR_Wavelet01   =   squeeze(dc_avg_wavlet(1:512,2,:))';
HbR_PCA99      =   squeeze(dc_avg_pca99(1:512,2,:))';
HbR_PCA97       =   squeeze(dc_avg_pca97(1:512,2,:))';
HbR_Cbsi        =   squeeze(dc_avg_cbsi(1:512,2,:))';
HbR_rloess        =   squeeze(dc_avg_rloess(1:512,2,:))';
%% real act
load('DAE/Real_NN_60208layers_act')

Hb_DAE = Y_real_act;
HbO_DAE = Hb_DAE(:, 1:512);
HbR_DAE = Hb_DAE(:, 513:end);
HbO_DAE = reshape(HbO_DAE',[],14)';
HbR_DAE = reshape(HbR_DAE',[],14)';

dc_HbO = HbO_DAE;
dc_HbR = HbR_DAE;
dc = [dc_HbO;dc_HbR]';


[dc_avg, ~, ~, ~, ~, ~] =   hmrBlockAvg(dc, s', t, [-39/fs_new (512-40)/fs_new] );
dc_DAE = dc_avg';
HbO_DAE = squeeze(dc_DAE(1:14,1:512));
HbR_DAE = squeeze(dc_DAE(15:end,1:512));

load('LSTM_AE/Real_NN_LSTM-AE')
Hb_LSTM_AE = Y_real_act;
HbO_LSTM_AE = Hb_LSTM_AE(:, 1:512);
HbR_LSTM_AE = Hb_LSTM_AE(:, 513:end);
HbO_LSTM_AE = reshape(HbO_LSTM_AE',[],14)';
HbR_LSTM_AE = reshape(HbR_LSTM_AE',[],14)';

dc_HbO = HbO_LSTM_AE;
dc_HbR = HbR_LSTM_AE;
dc = [dc_HbO;dc_HbR]';


[dc_avg, ~, ~, ~, ~, ~] =   hmrBlockAvg(dc, s', t, [-39/fs_new (512-40)/fs_new] );
dc_LSTM_AE = dc_avg';
HbO_LSTM_AE = squeeze(dc_LSTM_AE(1:14,1:512));
HbR_LSTM_AE= squeeze(dc_LSTM_AE(15:end,1:512));
%%
for channel = 1:size(HbO_real,1)
        HbO_mse_no_crct     =   [HbO_mse_no_crct; mean((HbO_no_crct(channel,:) - HbO_real(channel,:)).^2,2)*1e12];
        HbO_mse_Spline      =   [HbO_mse_Spline; mean((HbO_Spline(channel,:) - HbO_real(channel,:)).^2,2)*1e12];
        HbO_mse_Wavelet01   =   [HbO_mse_Wavelet01; mean((HbO_Wavelet01(channel,:) - HbO_real(channel,:)).^2,2)*1e12];
%         HbO_mse_PCA99      =   [HbO_mse_PCA99; mean((HbO_PCA99(channel,:) - HbO_real(channel,:)).^2,2)*1e12];
%         HbO_mse_PCA97       =   [HbO_mse_PCA97; mean((HbO_PCA97(channel,:) - HbO_real(channel,:)).^2,2)*1e12];
        HbO_mse_Cbsi        =   [HbO_mse_Cbsi; mean((HbO_Cbsi(channel,:) - HbO_real(channel,:)).^2,2)*1e12];
        HbO_mse_DAE          =   [HbO_mse_DAE; mean((HbO_DAE(channel,:) - HbO_real(channel,:)).^2,2)*1e12];
         HbO_mse_LSTM_AE          =   [HbO_mse_LSTM_AE; mean((HbO_LSTM_AE(channel,:) - HbO_real(channel,:)).^2,2)*1e12];
        HbO_mse_rloess          =   [HbO_mse_rloess; mean((HbO_rloess(channel,:) - HbO_real(channel,:)).^2,2)*1e12];

        HbR_mse_no_crct     =   [HbR_mse_no_crct; mean((HbR_no_crct(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
        HbR_mse_Spline      =   [HbR_mse_Spline; mean((HbR_Spline(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
        HbR_mse_Wavelet01   =   [HbR_mse_Wavelet01; mean((HbR_Wavelet01(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
        HbR_mse_PCA99      =   [HbR_mse_PCA99; mean((HbR_PCA99(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
        HbR_mse_PCA97       =   [HbR_mse_PCA97; mean((HbR_PCA97(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
        HbR_mse_Cbsi        =   [HbR_mse_Cbsi; mean((HbR_Cbsi(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
        HbR_mse_DAE          =   [HbR_mse_DAE; mean((HbR_DAE(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
        HbR_mse_LSTM_AE          =   [HbR_mse_LSTM_AE; mean((HbR_LSTM_AE(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
        HbR_mse_rloess          =   [HbR_mse_rloess; mean((HbR_rloess(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
end
labels = {'No correction','Spline','Wavelet01','PCA99','PCA797','Cbsi','DAE','rloess','LSTM_AE'};
%% sig test of mse
variables = {'no_crct', 'Spline', 'Wavelet01', 'Cbsi','rloess','DAE'};
fprintf('HbO:\n')
for i = 1:6
    x1 = HbO_mse_LSTM_AE;
    eval(strcat('x2 = HbO_mse_',variables{i},';'));
    delta_x = x2 -x1;
    h = kstest(delta_x);
    if h == 1
        p = signtest(delta_x);
    else
        p = ttest(delta_x);
    end
    p2 = ttest(delta_x);
    fprintf('For %s, h = %d; p = %.3f\n',variables{i}, h, p)
end
% fprintf('HbR:\n')
% for i = 1:8
%     x1 = HbR_mse_LSTM_AE;
%     eval(strcat('x2 = HbR_mse_',variables{i},';'));
%     delta_x = x2 - x1;
%     h = kstest(delta_x);
%     if h == 1
%         p = signtest(delta_x);
%     else
%         p = ttest(delta_x);
%     end
%     p2 = ttest(delta_x);
%     fprintf('For %s, h = %d; p = %.3f\n',variables{i}, h, p)
% end
fprintf('HbR:\n')
% for i = 1:6
%     x1 = HbR_mse_DAE;
%     eval(strcat('x2 = HbR_mse_',variables{i},';'));
%     delta_x = x2 - x1;
%     h = kstest(delta_x);
%     if h == 1
%         p = signtest(delta_x);
%     else
%         p = ttest(delta_x);
%     end
%     p2 = ttest(delta_x);
%     fprintf('For %s, h = %d; p = %.3f\n',variables{i}, h, p)
% end
