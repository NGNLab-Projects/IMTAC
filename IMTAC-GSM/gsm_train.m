clear all
clc
% Create environment
env = GSM_D3QN_Env();

% Get observation info
obsInfo = getObservationInfo(env);
obsDim = obsInfo.Dimension(1);

% Get action info (cell for multi-branch)
actInfo = getActionInfo(env);
numBranches = numel(actInfo);
branchSizes = zeros(1,numBranches);
for i = 1:numBranches
    branchSizes(i) = numel(actInfo(i).Elements);
end

% Create agent
agent = GSM_D3QN_AGENT(obsDim, branchSizes);

trainOpts = rlTrainingOptions( ...
    MaxEpisodes                = 20000, ...
    MaxStepsPerEpisode         = 50, ...
    ScoreAveragingWindowLength = 25, ...
    StopTrainingCriteria       = "EpisodeCount", ...
    StopTrainingValue          = 20000, ...
    SaveAgentCriteria          = "EpisodeFrequency", ...
    SaveAgentValue             = 100, ...         % save every 100 episodes
    SaveAgentDirectory         = "GSM_checkpoints", ...
    Verbose                    = true, ...
    Plots                      = "none");

% 3. Start your training
trainingStats = train(agent, env, trainOpts);

% Optional save
save("GSM_MCSandTAS_Aug19.mat","agent","trainingStats");

% ───────────────── TO RESUME FROM CHECKPOINT ─────────────────
% clear all
% clc
% 
% % Load the checkpoint — gives you saved_agent and savedAgentResult
% load("GSM_checkpoints/Agent2000.mat");   % adjust episode number as needed
% 
% % Sanity check
% fprintf('Resumed at episode    : %d\n',   savedAgentResult.EpisodeIndex(end));
% fprintf('Agent StepCount       : %d\n',   saved_agent.StepCount);
% fprintf('Epsilon               : %.6f\n', saved_agent.Epsilon);
% fprintf('Beta (PER)            : %.6f\n', saved_agent.Beta);
% 
% % Extend the episode budget in the embedded TrainingOptions
% savedAgentResult.TrainingOptions.MaxEpisodes = 4000;
% savedAgentResult.TrainingOptions.StopTrainingValue = 4000;   % must be > episodes already done
% 
% % Keep saving checkpoints during the resumed run
% savedAgentResult.TrainingOptions.SaveAgentCriteria  = "EpisodeFrequency";
% savedAgentResult.TrainingOptions.SaveAgentValue     = 100;
% savedAgentResult.TrainingOptions.SaveAgentDirectory = "GSM_checkpoints";
% 
% env = GSM_D3QN_Env();
% 
% % Native resume: pass savedAgentResult as 3rd arg — EpisodeIndex continues from 1001
% trainingStats = train(saved_agent, env, savedAgentResult);
% 
% save("GSM_MCSandTAS_June13_final2.mat", "saved_agent", "trainingStats");
