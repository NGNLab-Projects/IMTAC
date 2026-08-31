%% SM_D3QN_Inference.m
clear all
clc

%% ---------------- Config ----------------
AGENT_FILE      = 'SM_MCSandTAS_July25.mat';   % saved via save(...,"agent","trainingStats")
SNR_RANGE       = -20:1:35;      % dB, matches env's clamped SNR range
TRIALS_PER_SNR  = 10000;            % independent channel realizations per SNR point
SUMMARY_FILE    = 'SM_infer_summary_aug31.csv';   % per-SNR aggregate

SNR_MIN = -20; SNR_MAX = 35;

%% ---------------- Load agent ----------------
load(AGENT_FILE, 'agent');   % errors clearly if 'agent' var isn't in the file

agent.UseExplorationPolicy = false;   % greedy inference, epsilon irrelevant

%% ---------------- Environment ----------------
env = SM_D3QN_Env();

%% ---------------- Sweep ----------------
nSNR    = numel(SNR_RANGE);
results = table('Size',[nSNR*TRIALS_PER_SNR, 8], ...
    'VariableTypes', {'double','double','double','double','double','double','double','double'}, ...
    'VariableNames', {'SNR_dB','Trial','Alpha_cfgIdx','Tx_ants','Act_ants','MCS_ind','HARQ_ack','Reward'});

row = 1;
for si = 1:nSNR
    targetSNR = SNR_RANGE(si);

    for t = 1:TRIALS_PER_SNR
        % Fresh channel realization + CSI for this trial
        Observation = reset(env);

        % Override the random initial SNR with the sweep target
        env.SNR_dB          = targetSNR;
        env.normalized_snr  = 2*(targetSNR - SNR_MIN)/(SNR_MAX - SNR_MIN) - 1;
        Observation(1)       = env.normalized_snr;   % obs layout: [snr; harq; prevMCS; prevAlpha; csi...]

        % Greedy action from the trained policy
        action = getAction(agent, {Observation});
        if iscell(action)
            act = cell2mat(action);
        else
            act = action;
        end
        alpha = act(1); beta = act(2); gamma = act(3);

        cfg_idx  = alpha + 1;
        tx_ants  = env.VALID_CONFIGS(cfg_idx,1);
        act_ants = env.VALID_CONFIGS(cfg_idx,2);
        mcs_ind  = beta*5 + gamma;

        % Step (this also appends the full training-format row to
        % env.LogFilename via the class's own writematrix call)
        [ObsNext, Reward, ~, ~] = step(env, action);
        harq_ack = ObsNext(2);

        results(row,:) = {targetSNR, t, alpha, tx_ants, act_ants, mcs_ind, harq_ack, Reward};
        row = row + 1;
    end
end

writetable(results, 'SM_infer_pertrial_aug31.csv');

%% ---------------- Per-SNR aggregation ----------------
snrList   = SNR_RANGE(:);
aggTx     = zeros(nSNR,1);
aggAct    = zeros(nSNR,1);
aggMCS    = zeros(nSNR,1);
aggHARQ   = zeros(nSNR,1);
aggReward = zeros(nSNR,1);
aggTxStab = zeros(nSNR,1);   % fraction of trials matching the modal Tx config (policy stability)

for si = 1:nSNR
    mask = results.SNR_dB == SNR_RANGE(si);
    sub  = results(mask,:);

    aggTx(si)     = mode(sub.Tx_ants);
    aggAct(si)    = mode(sub.Act_ants);
    aggMCS(si)    = mode(sub.MCS_ind);
    aggHARQ(si)   = mean(sub.HARQ_ack);
    aggReward(si) = mean(sub.Reward);
    aggTxStab(si) = mean(sub.Tx_ants == mode(sub.Tx_ants));
end

summary = table(snrList, aggTx, aggAct, aggMCS, aggHARQ, aggReward, aggTxStab, ...
    'VariableNames', {'SNR_dB','ModalTxAnts','ModalActAnts','ModalMCS','MeanHARQ_ack','MeanReward','TxConfigStability'});

writetable(summary, SUMMARY_FILE);

%% ---------------- Plots ----------------
figure('Name','SM D3QN Inference — Policy vs SNR');

subplot(4,1,1);
plot(snrList, aggTx, '-o', snrList, aggAct, '-s'); grid on;
legend('Tx antennas (Nt)','Active antennas (Na)','Location','best');
ylabel('Antennas'); title('Chosen antenna configuration vs SNR');

subplot(4,1,2);
plot(snrList, aggMCS, '-o'); grid on;
ylabel('MCS index (0-24)'); title('Chosen MCS vs SNR');

subplot(4,1,3);
plot(snrList, aggHARQ, '-o'); grid on; ylim([0 1.05]);
ylabel('HARQ ACK rate'); title('Mean HARQ success rate vs SNR');

subplot(4,1,4);
plot(snrList, aggReward, '-o'); grid on;
ylabel('Reward'); xlabel('SNR (dB)'); title('Mean reward vs SNR');

sgtitle('SM D3QN — Greedy Policy Evaluation');
saveas(gcf, 'SM_inference_policy_vs_snr.png');

fprintf('Done. Per-trial log: SM_inference_pertrial.csv | Summary: %s | Raw step log: %s (see env.LogFilename)\n', ...
    SUMMARY_FILE, env.LogFilename);