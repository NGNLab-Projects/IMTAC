classdef GSM_D3QN_AGENT < rl.agent.CustomAgent
  properties
      NumObs
      BranchSizes
      Gamma = 0.99
      Epsilon = 1
      LearnRate = 1e-4
      MiniBatchSize = 128
      TargetSmoothFactor = 0.005
      BufferSize = 2e5
      EpsilonLog = []
      EpsilonLogInterval = 100
      StepCount = 0
      EpsilonDecaySteps = 500000  % decay from step 5000 to step 20000
      ReplayWarmup = 20000
     
      % PER Hyperparameters
      Alpha = 0.4         
      Beta = 0.6         
      BetaGrowth = 5e-7   
  end
 
  properties (Access = private)
      OnlineNet
      TargetNet
      ReplayBuffer
      Priorities 
      BufferIndex = 1
      BufferCount = 0
      Velocity = []
  end

   methods
      function obj = GSM_D3QN_AGENT(obsDim, branchSizes)
          obj = obj@rl.agent.CustomAgent();
          obj.NumObs = obsDim;
          obj.BranchSizes = branchSizes;
          obj.ObservationInfo_ = rlNumericSpec([obsDim 1]);
        
          obj.ActionInfo_ = rl.util.rlFiniteSetSpec.empty(0, 1);
          for i = 1:numel(branchSizes)
              obj.ActionInfo_(i,1) = rlFiniteSetSpec(0:branchSizes(i)-1); % 0-indexed
          end
        
          obj.OnlineNet = obj.createNetwork();
          obj.OnlineNet = initialize(obj.OnlineNet);
          obj.TargetNet = obj.OnlineNet;
         
          obj.ReplayBuffer = cell(obj.BufferSize,1);
          obj.Priorities = zeros(obj.BufferSize, 1);
      end
      function net = getOnlineNet(obj)
          net = obj.OnlineNet;
      end
   end
   
  methods (Access = protected)
      function action = getActionImpl(obj,obs)
            Q = obj.computeQ(obj.OnlineNet, obs);
            action = cell(numel(obj.BranchSizes),1);
            for i = 1:numel(obj.BranchSizes)
                [~,idx] = max(Q{i});
                action{i} = double(idx(1)) - 1;
            end
            obj.logStateActionQ(obs, Q, action, false);
        end
        
        function action = getActionWithExplorationImpl(obj,obs)
            Q = obj.computeQ(obj.OnlineNet, obs);
            isExploratory = rand < obj.Epsilon;
        
            action = cell(numel(obj.BranchSizes),1);
            for i = 1:numel(obj.BranchSizes)
                if isExploratory
                    action{i} = double(randi(obj.BranchSizes(i)) - 1);
                else
                    [~,idx] = max(Q{i});
                    action{i} = double(idx(1)) - 1;
                end
            end
        
            obj.logStateActionQ(obs, Q, action, isExploratory);
        end
      function action = learnImpl(obj,exp)       
        
          obj.StepCount = obj.StepCount + 1;
          obj.Epsilon = max(0, 1 - max(0, obj.StepCount - obj.ReplayWarmup) / obj.EpsilonDecaySteps); 
          obj.ReplayBuffer{obj.BufferIndex} = exp;       
        
            if obj.BufferCount == 0
                newPriority = 1.0;
            else
                newPriority = max(obj.Priorities(1:obj.BufferCount), [], 'all');
            end      
        
            obj.Priorities(obj.BufferIndex) = newPriority;        
        
            obj.BufferIndex = mod(obj.BufferIndex, obj.BufferSize) + 1;
            obj.BufferCount = min(obj.BufferCount + 1, obj.BufferSize);      
        
            % ---- Replay warm-up before learning ----      
            if obj.BufferCount >= obj.ReplayWarmup
                batchIdx = obj.sampleBatchIdx();
                obj.updateNetwork(batchIdx);      
                obj.softUpdateTarget();
            end 
            obj.EpsilonLog(end+1) = obj.Epsilon; % Log Epsilon
            if numel(obj.EpsilonLog) >= obj.EpsilonLogInterval
                writematrix(obj.EpsilonLog(:), 'GSM_agent_epsilon_aug19.csv', 'WriteMode', 'append');
                obj.EpsilonLog = [];
            end
            action = getActionWithExplorationImpl(obj, exp{4});        
        end
  end

  methods (Access = private)
      function net = createNetwork(obj)
            % -------- Shared representation: 7 hidden layers --------
            sharedLayers = [
                featureInputLayer(obj.NumObs, 'Normalization', 'none', 'Name', 'state')
                fullyConnectedLayer(256, 'Name', 'shared_fc1')
                reluLayer('Name', 'shared_relu1')
                fullyConnectedLayer(256, 'Name', 'shared_fc2')
                reluLayer('Name', 'shared_relu2')
                fullyConnectedLayer(256, 'Name', 'shared_fc3')
                reluLayer('Name', 'shared_relu3')
                fullyConnectedLayer(256, 'Name', 'shared_fc4')
                reluLayer('Name', 'shared_relu4')
                fullyConnectedLayer(256, 'Name', 'shared_fc5')
                reluLayer('Name', 'shared_relu5')
                fullyConnectedLayer(256, 'Name', 'shared_fc6')
                reluLayer('Name', 'shared_relu6')
                fullyConnectedLayer(256, 'Name', 'shared_fc7')
                reluLayer('Name', 'shared_relu7')
            ];
            lg = layerGraph(sharedLayers);
        
            % -------- Value stream: 1 hidden layer --------
            valueLayers = [
                fullyConnectedLayer(128, 'Name', 'V_fc1')
                reluLayer('Name', 'V_relu1')
                fullyConnectedLayer(1, 'Name', 'V')
            ];
            lg = addLayers(lg, valueLayers);
            lg = connectLayers(lg, 'shared_relu7', 'V_fc1');
        
            % -------- Advantage streams: 3 hidden layers each --------
            for i = 1:numel(obj.BranchSizes)
                advLayers = [
                    fullyConnectedLayer(128, 'Name', "A"+i+"_fc1")
                    reluLayer('Name', "A"+i+"_relu1")
                    fullyConnectedLayer(128, 'Name', "A"+i+"_fc2")
                    reluLayer('Name', "A"+i+"_relu2")
                    fullyConnectedLayer(128, 'Name', "A"+i+"_fc3")
                    reluLayer('Name', "A"+i+"_relu3")
                    fullyConnectedLayer(obj.BranchSizes(i), 'Name', "A"+i)
                ];
                lg = addLayers(lg, advLayers);
                lg = connectLayers(lg, 'shared_relu7', "A"+i+"_fc1");
            end
        
            net = dlnetwork(lg);
        end
      function Q = computeQ(obj,net,x)
          if iscell(x), x = x{1}; end
          if size(x,1) ~= obj.NumObs, x = x'; end
          dlX = dlarray(double(x),"CB");
          Q_dl = obj.computeQ_dl(net,dlX);
          Q = cell(numel(obj.BranchSizes),1);
          for i = 1:numel(obj.BranchSizes), Q{i} = extractdata(Q_dl{i}); end
      end
      function Q = computeQ_dl(obj,net,dlX)
          outNames = ["V", "A" + (1:numel(obj.BranchSizes))];
          out = cell(1, numel(outNames));
          [out{:}] = forward(net, dlX, 'Outputs', outNames);
          V = out{1};
          Q = cell(numel(obj.BranchSizes),1);
          for i = 1:numel(obj.BranchSizes)
              A = out{1+i};
              Q{i} = V + (A - mean(A, 1));
          end
      end
      function idx = sampleBatchIdx(obj)
          p = obj.Priorities(1:obj.BufferCount);
          p = (p + 1e-6).^obj.Alpha;
          probs = p / sum(p);
          idx = randsample(obj.BufferCount, obj.MiniBatchSize, true, probs);
      end
      function updateNetwork(obj, batchIdx)       
            bs = obj.MiniBatchSize;
            batch = obj.ReplayBuffer(batchIdx);        
            % Vectorization
            S = zeros(obj.NumObs, bs);
            S2 = zeros(obj.NumObs, bs);
            R = zeros(1, bs);
            D = zeros(1, bs);
            numActions = numel(obj.BranchSizes);
            A = zeros(numActions, bs);      
            for i = 1:bs
                e = batch{i};
                S(:,i) = e{1}{1};     
                act = e{2};
                if iscell(act), act = cell2mat(act); end
                A(:,i) = act;
                R(i) = e{3};
                S2(:,i) = e{4}{1};
                D(i) = e{5};
            end
        
            % Double DQN target computation
            Q_next_online = obj.computeQ(obj.OnlineNet, S2);
            Q_next_target = obj.computeQ(obj.TargetNet, S2);
        
            mean_target_next = zeros(1, bs);
            for k = 1:numActions
                [~, aMax] = max(Q_next_online{k}, [], 1);
                idx_target = sub2ind(size(Q_next_target{k}), aMax, 1:bs);
                mean_target_next = mean_target_next + Q_next_target{k}(idx_target);
            end
            mean_target_next = mean_target_next / numActions;
            
            shared_target = R + obj.Gamma * (1-D) .* mean_target_next;
            
            targets = cell(numActions, 1);
            for k = 1:numActions
                targets{k} = shared_target;  % identical for all branches
            end
            % ----- Correct PER importance sampling weights ----- 
            p_all = obj.Priorities(1:obj.BufferCount);
            p_all = (p_all + 1e-6).^obj.Alpha;  
            probs = p_all ./ sum(p_all);  
            p_batch = probs(batchIdx);
            weights = (obj.BufferCount .* p_batch) .^ (-obj.Beta);
            weights = weights ./ max(weights);
            % ----- Gradient computation ------
            [grad, ~, tdErrors] = dlfeval(@obj.modelGradients, obj.OnlineNet, S, A, targets, weights);    
            % ----- Priority update (with clipping for stability) -----    
            tdErrors = abs(tdErrors);
            tdErrors = min(tdErrors, 10);   % clip extreme TD errors  
            obj.Priorities(batchIdx) = tdErrors(:) + 1e-6;       
            % ----- Network update -----
            Momentum = 0.9;  % standard SGD momentum value
            [obj.OnlineNet.Learnables, obj.Velocity] = sgdmupdate( ...
                obj.OnlineNet.Learnables, grad, obj.Velocity, ...
                obj.LearnRate, Momentum);  
            % ----- Anneal PER beta -----
            obj.Beta = min(1, obj.Beta + obj.BetaGrowth);   
        end

      function [grad, loss, batchErrors] = modelGradients(obj, net, S, A, targets, weights)
        
            dlS = dlarray(S,"CB");
            Q = obj.computeQ_dl(net, dlS);
            batchLoss = dlarray(zeros(1, size(S,2)));
            batchErrors = dlarray(zeros(1, size(S,2)));        
            for i = 1:numel(obj.BranchSizes)
                Qk = Q{i};
                idx = sub2ind(size(Qk), A(i,:) + 1, 1:size(S,2));     
                td = targets{i} - Qk(idx);
        
                % Loss for training
                batchLoss = batchLoss + (td.^2);
        
                % Priority signal for PER
                batchErrors = batchErrors + abs(td);   
            end
            batchLoss = batchLoss / numel(obj.BranchSizes);  
            weightedLoss = dlarray(weights(:)' ,"CB") .* batchLoss; 

            % Mean loss for stable gradients
            loss = mean(weightedLoss,'all');
            grad = dlgradient(loss, net.Learnables);
            rescale = 1 / (numel(obj.BranchSizes) + 1);
            layerNames = grad.Properties.RowNames;
            numGradRows = numel(layerNames);
            for gi = 1:numGradRows
                if contains(layerNames{gi}, 'shared')
                    if ~isempty(grad.Value{gi})
                        grad.Value{gi} = extractdata(grad.Value{gi}) * rescale;
                    end
                end
            end
        
            % Return PER priority values
            batchErrors = extractdata(batchErrors);        
      end

      function softUpdateTarget(obj)
          tau = obj.TargetSmoothFactor;
          obj.TargetNet.Learnables = dlupdate(@(t,o) (1-tau)*t + tau*o, ...
              obj.TargetNet.Learnables, obj.OnlineNet.Learnables);
      end

      function logStateActionQ(obj, obs, Q, action, isExploratory)
          if iscell(obs), obs = obs{1}; end
          obs = double(obs(:)');

          Qflat = [];
          for i = 1:numel(obj.BranchSizes)
              Qflat = [Qflat, Q{i}(:)']; %#ok<AGROW>
          end

          actFlat = cellfun(@double, action(:)');

          row = [obs, Qflat, actFlat, double(isExploratory), obj.StepCount];
          writematrix(row, 'GSM_state_action_q_aug19.csv', 'WriteMode', 'append');
      end
  end
end

