%%对模拟数据的mse性能分析
%% 对于6020nirs5受试者的分析
  define_constants
%% 对模拟的数据进行mse分析
load('C:\Users\yp181\Desktop\Processed_data\SimulateData.mat','HRF_test_noised')
[m,n] = size(HRF_test_noised);
HbO_noised = HRF_test_noised(1:m/2,:);
HbR_noised = HRF_test_noised(m/2+1:end,:);

%% 配置sd文件
    SD1.MeasList = [1,1,1,1;1,1,1,2];
    SD1.MeasListAct = [1 1];
    SD1.Lambda = [760;850];
    SD1.SrcPos = [-2.9017 10.2470 -0.4494];
    SD1.DetPos = [-4.5144 9.0228 -1.6928];
    ppf = [1,1];
    t  = 1/fs_new:1/fs_new:size(HbO_noised,2)/fs_new;
    s  = zeros(1,length(t));
    s((rt):512:length(t)) = 1;
    tIncMan=ones(size(t))';
    SD = SD1;
    %% Cbsi
    HbO_Cbsi    = [];
    HbR_Cbsi    = [];
    n_Cbsi = 0;
    T_Cbsi = 0;

    for i = 1:m/2
        dc_HbO          =   HbO_noised(i,:);
        dc_HbR          =   HbR_noised(i,:);
        dc              =   [dc_HbO;dc_HbR]';

        tic
        [dc_avg,n_MA]   =   proc_Cbsi(dc,s,SD,t,tIncMan,OD_thred,STD);
        T_Cbsi          =   T_Cbsi + toc;

        HbO_Cbsi(end+1,:) = dc_avg(:,1)';
        HbR_Cbsi(end+1,:) = dc_avg(:,2)';
        n_Cbsi          =   n_Cbsi + n_MA;
    end
    save_path = fullfile('Processed_data/process_simdata', 'Testing_Cbsi.mat');
    save(save_path,'HbO_Cbsi','HbR_Cbsi','n_Cbsi', 'T_Cbsi')
    %% PCA97
    HbO_PCA97   = [];
    HbR_PCA97   = [];
    n_PCA97 = 0;
    sigma   =  0.97;
    T_PCA97 =   0;

    for i = 1:m/2
        dc_HbO          =   HbO_noised(i,:);
        dc_HbR          =   HbR_noised(i,:);
        dc              =   [dc_HbO;dc_HbR]';

        tic
        [dc_avg,n_MA]   =   proc_PCA(dc, s, SD, t, tIncMan,STD, OD_thred, sigma);
        T_PCA97         =   T_PCA97 + toc;

        HbO_PCA97(end+1,:)  = dc_avg(:,1)';
        HbR_PCA97(end+1,:)  = dc_avg(:,2)';
        n_PCA97             =   n_PCA97 + n_MA;
    end
    save_path = fullfile('Processed_data/process_simdata', 'Testing_PCA97.mat');
    save(save_path,'HbO_PCA97', 'HbR_PCA97', 'n_PCA97', 'T_PCA97')
    %% PCA99
    HbO_PCA99   = [];
    HbR_PCA99   = [];
    n_PCA99 = 0;
    sigma   =  0.99;
    T_PCA99 =   0;

    for i = 1:m/2
        dc_HbO          =   HbO_noised(i,:);
        dc_HbR          =   HbR_noised(i,:);
        dc              =   [dc_HbO;dc_HbR]';

        tic
        [dc_avg,n_MA]   =   proc_PCA(dc, s, SD, t, tIncMan,STD, OD_thred, sigma);
        T_PCA99         =   T_PCA99 + toc;

        HbO_PCA99(end+1,:)  = dc_avg(:,1)';
        HbR_PCA99(end+1,:)  = dc_avg(:,2)';
        n_PCA99             =   n_PCA99 + n_MA;
    end
    save_path = fullfile('Processed_data/process_simdata', 'Testing_PCA99.mat');
    save(save_path,'HbO_PCA99','HbR_PCA99','n_PCA99')
    %% Spline
    HbO_Spline  = [];
    HbR_Spline  = [];
    n_Spline = 0;
    T_Spline =   0;
    p       =   0.99;

    for i = 1:m/2
        dc_HbO              =   HbO_noised(i,:);
        dc_HbR              =   HbR_noised(i,:);
        dc                  =   [dc_HbO;dc_HbR]';

        tic
        [dc_avg,n_MA]       =   proc_Spline(dc, s, SD, t, tIncMan, STD, OD_thred,p);
        T_Spline            =   T_Spline + toc;

        HbO_Spline(end+1,:)  = dc_avg(:,1)';
        HbR_Spline(end+1,:)  = dc_avg(:,2)';
        n_Spline             =   n_Spline + n_MA;
    end
    save_path = fullfile('Processed_data/process_simdata', 'Testing_Spline.mat');
    save(save_path,'HbO_Spline', 'HbR_Spline', 'n_Spline', 'T_Spline')
    %% Wavelet01
    HbO_Wavelet01 = [];
    HbR_Wavelet01 = [];
    n_Wavelet01 = 0;
    T_Wavelet01 =   0;
    iqr       =   0.75;

    for i = 1:m/2
        dc_HbO              =   HbO_noised(i,:);
        dc_HbR              =   HbR_noised(i,:);
        dc                  =   [dc_HbO;dc_HbR]';

        tic
        [dc_avg,n_MA]       =   proc_Wavelet(dc,s,SD1,t,tIncMan,STD,OD_thred,iqr);
        T_Wavelet01         =   T_Wavelet01 + toc;

        HbO_Wavelet01(end+1,:)  = dc_avg(:,1)';
        HbR_Wavelet01(end+1,:)  = dc_avg(:,2)';
        n_Wavelet01             =   n_Wavelet01 + n_MA;
    end
    save_path = fullfile('Processed_data/process_simdata', 'Testing_Wavelet01.mat');
     save(save_path,'HbO_Wavelet01','HbR_Wavelet01','n_Wavelet01', 'T_Wavelet01')
     %% rloess
      HbO_rloess = [];
    HbR_rloess= [];
    n_rloess = 0;
    T_rloess =   0;
    for i = 1:m/2
        dc_HbO              =   HbO_noised(i,:);
        dc_HbR              =   HbR_noised(i,:);
        dc                  =   [dc_HbO;dc_HbR]';

        tic
        [dc_avg,n_MA]       =   proc_rloess(dc, s, SD1, t, tIncMan, STD, OD_thred);
        T_rloess         =   T_rloess + toc;

        HbO_rloess(end+1,:)  = dc_avg(:,1)';
        HbR_rloess(end+1,:)  = dc_avg(:,2)';
        n_rloess             =   n_rloess + n_MA;
    end
    save_path = fullfile('Processed_data/process_simdata', 'Testing_rloess.mat');
     save(save_path,'HbO_rloess','HbR_rloess','n_rloess', 'T_rloess')
    %% no correction
    n_no_correction = 0;

    for i = 1:m/2
        dc_HbO              =   HbO_noised(i,:);
        dc_HbR              =   HbR_noised(i,:);
        dc                  =   [dc_HbO;dc_HbR]';
        dod                 =   hmrConc2OD( dc, SD1, ppf );
        [~,tIncAuto]            =   hmrMotionArtifactByChannel(dod,t,SD1,tIncMan,0.5,1,STD,OD_thred);
        n_MA                =   count_MA(tIncAuto);

        n_no_correction     =   n_no_correction + n_MA;
    end
    save_path = fullfile('Processed_data/process_simdata', 'Testing_no_correction.mat');
     save(save_path,'n_no_correction')
    %% NN count noise
    load('Test_NN_6020_ma8layers.mat')% File exists.
    Hb_NN = Y_test;
    m = size(Hb_NN,1);

    HbO_NN = Hb_NN(:,1:512);
    HbR_NN = Hb_NN(:,512+1:end);
    % HbO_NN = reshape(HbO_NN',[512*5,1684]);
    % HbR_NN = reshape(HbR_NN',[512*5,1684]);
    % HbO_NN = HbO_NN';
    % HbR_NN = HbR_NN';
    n_NN = 0;
    t = t(1:512);
    tIncMan = tIncMan(1:512);

    for i = 1:m
        dc_HbO  =   HbO_NN(i,:);
        dc_HbR  =   HbR_NN(i,:);
        dc      =   [dc_HbO;dc_HbR]';
        dod     =   hmrConc2OD( dc, SD1, ppf );
        [~,tIncAuto]            =   hmrMotionArtifactByChannel(dod,t,SD,tIncMan,0.5,1,STD,OD_thred);
    %     tIncAuto=   hmrMotionArtifact(dod,t,SD1,tIncMan,0.5,1,STD,OD_thred);
        n_MA    =   count_MA(tIncAuto);

        n_NN = n_NN + n_MA;
    end
     save_path = fullfile('Processed_data/process_simdata', 'Testing_NN.mat');
    save(save_path,'n_NN')
    
    %% mse
    HbO_mse_no_crct     =   [];
    HbO_mse_Spline      =   [];
    HbO_mse_Wavelet01   =   [];
    HbO_mse_PCA97       =   [];
    HbO_mse_Cbsi        =   [];
    HbO_mse_DAE          =   [];
    HbO_mse_LSTM_AE          =   [];
    
    HbR_mse_no_crct     =   [];
    HbR_mse_Spline      =   [];
    HbR_mse_Wavelet01   =   [];
    HbR_mse_PCA97       =   [];
    HbR_mse_Cbsi        =   [];
    HbR_mse_DAE          =   [];
    HbR_mse_LSTM_AE          =   [];
    
    load('Processed_data\process_simdata\Testing_Spline.mat')
    load('Processed_data\process_simdata\Testing_Wavelet01.mat')
    load('Processed_data\process_simdata\Testing_PCA97.mat')
    load('Processed_data\process_simdata\Testing_PCA99.mat')
    load('Processed_data\process_simdata\Testing_Cbsi.mat')
    load('Processed_data\process_simdata\Testing_rloess.mat')
    load('Processed_data\leave_3_out\SimulateData')
    %%
    [m,n] = size(HRF_test_noised);
    HbO_test_noised = HRF_test_noised(1:m/2,:);
    HbO_test = HRF_test(1:m/2,:);
    HbR_test_noised = HRF_test_noised(m/2+1:end,:);
    HbR_test = HRF_test(m/2+1:end,:);
    HbO_no_crct = zeros(size(HbO_Cbsi));
    HbR_no_crct = zeros(size(HbO_Cbsi));
    HbO_real = zeros(size(HbO_Cbsi));
    HbR_real = zeros(size(HbO_Cbsi));
    
    define_constants

    t  = 1/fs_new:1/fs_new:size(HbO_test_noised,2)/fs_new;
    s  = zeros(1,length(t));
    s((rt):512:length(t)) = 1;
    tIncMan=ones(size(t))';

    for i = 1:size(HbO_test,1)
        dc_HbO  = HbO_test_noised(i,:);
        dc_HbR  = HbR_test_noised(i,:);
        dc      =   [dc_HbO;dc_HbR]';
        [dc_avg, ~, ~, ~, ~, ~] =   hmrBlockAvg(dc, s', t, [-39/fs_new (512-40)/fs_new] ); 
        HbO_no_crct(i,:) = dc_avg(:,1)';
        HbR_no_crct(i,:) = dc_avg(:,2)';

        HbO_real(i,:) = HbO_test(i,1:512);
        HbR_real(i,:) = HbR_test(i,1:512);
    end
    %%
    load('DAE/Test_NN_60208layers')
    HbO_DAE = zeros(size(HbO_Cbsi));
    HbR_DAE = zeros(size(HbO_Cbsi));
    j = 1;
    for i = 1:5:size(Y_test,1)
        HbO_DAE(j,:) = mean(Y_test(i:i+4,1:512),1);
        HbR_DAE(j,:) = mean(Y_test(i:i+4,513:end),1);
        j = j + 1;
    end
    
     load('LSTM_AE/Test_NN_LSTM-AE')
    HbO_LSTM_AE = zeros(size(HbO_Cbsi));
    HbR_LSTM_AE = zeros(size(HbO_Cbsi));
    j = 1;
    for i = 1:5:size(Y_test,1)
        HbO_LSTM_AE(j,:) = mean(Y_test(i:i+4,1:512),1);
        HbR_LSTM_AE(j,:) = mean(Y_test(i:i+4,513:end),1);
        j = j + 1;
    end
    %%
    HbO_mse_no_crct     =   [HbO_mse_no_crct; mean((HbO_no_crct - HbO_real).^2,2)*1e12];
    HbO_mse_Spline      =   [HbO_mse_Spline; mean((HbO_Spline - HbO_real).^2,2)*1e12];
    HbO_mse_Wavelet01   =   [HbO_mse_Wavelet01; mean((HbO_Wavelet01 - HbO_real).^2,2)*1e12];
    HbO_mse_PCA97      =   [HbO_mse_PCA97; mean((HbO_PCA97 - HbO_real).^2,2)*1e12];
    HbO_mse_Cbsi        =   [HbO_mse_Cbsi; mean((HbO_Cbsi - HbO_real).^2,2)*1e12];
    HbO_mse_DAE          =   [HbO_mse_DAE; mean((HbO_DAE - HbO_real).^2,2)*1e12];
    HbO_mse_LSTM_AE          =   [HbO_mse_LSTM_AE; mean((HbO_LSTM_AE - HbO_real).^2,2)*1e12];

    HbR_mse_no_crct     =   [HbR_mse_no_crct; mean((HbR_no_crct - HbR_real).^2,2)*1e12];
    HbR_mse_Spline      =   [HbR_mse_Spline; mean((HbR_Spline - HbR_real).^2,2)*1e12];
    HbR_mse_Wavelet01   =   [HbR_mse_Wavelet01; mean((HbR_Wavelet01 - HbR_real).^2,2)*1e12];
    HbR_mse_PCA97      =   [HbR_mse_PCA97; mean((HbR_PCA97 - HbR_real).^2,2)*1e12];
    HbR_mse_Cbsi        =   [HbR_mse_Cbsi; mean((HbR_Cbsi - HbR_real).^2,2)*1e12];
    HbR_mse_DAE          =   [HbR_mse_DAE; mean((HbR_DAE - HbR_real).^2,2)*1e12];
    HbR_mse_LSTM_AE          =   [HbR_mse_LSTM_AE; mean((HbR_LSTM_AE - HbR_real).^2,2)*1e12];

%% t分析
variables = {'no_crct','Spline', 'Wavelet01',  'PCA97', 'Cbsi','DAE'};
fprintf('HbO:\n')
for i = 1:6
    i = 6;
    x1 = HbO_mse_LSTM_AE;
    eval(strcat('x2 = HbO_mse_',variables{i},';'));
    delta_x = x2 - x1;
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
% for i = 1:5
%     x1 = HbR_mse_NN;
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

  