classdef SM_D3QN_Env < rl.env.MATLABEnvironment

    properties
        % ── Hardcoded GSM config table ─────────────────────────────────────
        % Each row is one allowed [Nt, Na] pair.
        % alpha action (0-based) indexes into this table: alpha=0 → row 1, etc.
        VALID_CONFIGS = [
             2,  1;
             4,  1;
             8,  1;
            16,  1;
            32,  1;];

        tx_ants           % Nt: set from chosen config at each step
        act_ants          % Na: set from chosen config at each step
        combo_cache 
        SNR_dB
        harq_ack_nack
        raw_channel
        local_channel
        csi_vector
        prev_mcs_ind     
        prev_alpha
        normalized_snr
    end

    properties(Access = protected)
        IsDone = false
    end

    methods
        function this = SM_D3QN_Env()
            ObservationInfo = rlNumericSpec([260 1]);
            ObservationInfo.Name = 'Wireless Observation';

            % Act1: config selection — 5 allowed (Nt,Na) pairs (0-indexed)
            % Act2: MCS tens digit  — beta  in {0,1,2,3,4}
            % Act3: MCS units digit — gamma in {0,1,2,3,4}
            % mcs_ind = beta*5 + gamma  →  range [0,24]  (25 MCS entries)
            Act1 = rlFiniteSetSpec(0:4);   % 5 combinations
            Act2 = rlFiniteSetSpec(0:4);   % beta
            Act3 = rlFiniteSetSpec(0:4);   % gamma
            ActionInfo = [Act1; Act2; Act3];

            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
            reset(this);
        end

        function [Observation,Reward,IsDone,Info] = step(this,Action)
            Info = [];

            if iscell(Action)
                Action = cell2mat(Action);
            end

            % ── Action decoding ────────────────────────────────────────────
            alpha = Action(1);   % config index, 0-based → rows of VALID_CONFIGS
            beta  = Action(2);   % MCS tens digit
            gamma = Action(3);   % MCS units digit

            % GSM config: alpha selects (Nt,Na) pair from hardcoded table
            cfg_idx       = alpha + 1;                          % 1-based row index
            this.tx_ants  = this.VALID_CONFIGS(cfg_idx, 1);    % Nt
            this.act_ants = this.VALID_CONFIGS(cfg_idx, 2);    % Na

            % MCS index (0-based, into mcs_vals from mcstable3gpp)
            mcs_ind = beta * 5 + gamma;   % range [0,24]

            % ── Local aliases ──────────────────────────────────────────────
            tx_ants  = this.tx_ants;
            act_ants = this.act_ants;

            % ── GSM combination table for chosen config ────────────────────
            all_combos      = this.combo_cache{cfg_idx};
            num_combos      = nchoosek(tx_ants, act_ants);
            gsm_idx_bits    = floor(log2(num_combos));
            num_used_combos = 2^gsm_idx_bits;

            % ── Simulation parameters ──────────────────────────────────────
            simParameters = struct();
            simParameters.NFrames = 1;
            simParameters.NTxAnts = 1;
            simParameters.NRxAnts = 4;
            simParameters.DataType = 'single';
            simParameters.Prototype = zeros(0,0,simParameters.DataType);

            simParameters.Carrier = nrCarrierConfig;
            simParameters.Carrier.NSizeGrid = 51;
            simParameters.Carrier.SubcarrierSpacing = 30;
            simParameters.Carrier.CyclicPrefix = 'Normal';
            simParameters.Carrier.NCellID = 1;

            simParameters.PDSCH = nrPDSCHConfig;
            simParameters.PDSCHExtension = struct();

            simParameters.PDSCH.PRBSet = 0:simParameters.Carrier.NSizeGrid-1;
            simParameters.PDSCH.SymbolAllocation = [0,simParameters.Carrier.SymbolsPerSlot];
            simParameters.PDSCH.MappingType = 'A';
            simParameters.PDSCH.NID = simParameters.Carrier.NCellID;
            simParameters.PDSCH.RNTI = 1;
            simParameters.PDSCH.VRBToPRBInterleaving = 0;
            simParameters.PDSCH.VRBBundleSize = 4;
            simParameters.PDSCH.DMRS.DMRSTypeAPosition = 2;
            simParameters.PDSCH.DMRS.DMRSLength = 1;
            simParameters.PDSCH.DMRS.DMRSAdditionalPosition = 2;
            simParameters.PDSCH.DMRS.DMRSConfigurationType = 2;
            simParameters.PDSCH.DMRS.NumCDMGroupsWithoutData = 1;
            simParameters.PDSCH.DMRS.NIDNSCID = 1;
            simParameters.PDSCH.DMRS.NSCID = 0;

            simParameters.PDSCHExtension.PRGBundleSize = [];
            simParameters.PDSCHExtension.XOverhead = 0;
            simParameters.PDSCHExtension.NHARQProcesses = 5;
            simParameters.PDSCHExtension.EnableHARQ = true;
            simParameters.PDSCHExtension.EnableCBGTransmission = false;
            simParameters.PDSCHExtension.MaxNumCBG = 4;

            rvSeq = [0 2 3 1];

            simParameters.PDSCHExtension.LDPCDecodingAlgorithm = 'Normalized min-sum';
            simParameters.PDSCHExtension.MaximumLDPCIterationCount = 2;

            waveformInfo = nrOFDMInfo(simParameters.Carrier);

            mcstable3gpp
            mod_scheme = mcs_vals{mcs_ind+1,2};
            code_rate  = mcs_vals{mcs_ind+1,3};
            simParameters.SNRIn = this.SNR_dB;
            modOrderMap = containers.Map({'BPSK','QPSK','16QAM','64QAM','256QAM'}, {1,2,4,6,8});
            Qm = modOrderMap(mod_scheme);

            simParameters.PDSCH.NumLayers = 1;
            simParameters.PDSCH.Modulation = mod_scheme;
            simParameters.PDSCHExtension.TargetCodeRate = code_rate;
            simParameters.PDSCH.DMRS.DMRSPortSet = 0:simParameters.PDSCH.NumLayers-1;

            % ────────────────── Channel ────────────────────────────────────
            % raw_channel was generated for Nt=32 in reset().
            % Slice to the Nt columns needed by the chosen config.
            channel     = this.local_channel;
            chInfo      = info(channel);
            maxChDelay  = chInfo.MaximumChannelDelay;

            maxThroughput = zeros(length(simParameters.SNRIn),1);
            simThroughput = zeros(length(simParameters.SNRIn),1);
            % totalAntBitErrors = zeros(length(simParameters.SNRIn), 1);
            % totalAntBits      = zeros(length(simParameters.SNRIn), 1);

            encodeDLSCH = nrDLSCH;
            encodeDLSCH.MultipleHARQProcesses = true;
            encodeDLSCH.CBGTransmission = true;
            encodeDLSCH.TargetCodeRate = simParameters.PDSCHExtension.TargetCodeRate;

            decodeDLSCH = nrDLSCHDecoder;
            decodeDLSCH.MultipleHARQProcesses = true;
            decodeDLSCH.CBGTransmission = true;
            decodeDLSCH.TargetCodeRate = simParameters.PDSCHExtension.TargetCodeRate;
            decodeDLSCH.LDPCDecodingAlgorithm = simParameters.PDSCHExtension.LDPCDecodingAlgorithm;
            decodeDLSCH.MaximumLDPCIterationCount = simParameters.PDSCHExtension.MaximumLDPCIterationCount;

            simLocal        = simParameters;
            waveinfoLocal   = waveformInfo;
            carrier         = simLocal.Carrier;
            pdsch           = simLocal.PDSCH;
            pdschextra      = simLocal.PDSCHExtension;
            decodeDLSCHLocal = decodeDLSCH;
            decodeDLSCHLocal.reset();

            SNRdB = simLocal.SNRIn;
            harqSequence = 0:pdschextra.NHARQProcesses-1;
            maxNumCBG = 1;
            harqEntity = HARQEntity(harqSequence,rvSeq,pdsch.NumCodewords,maxNumCBG);

            NSlots = simLocal.NFrames * carrier.SlotsPerFrame;

            harqvals = zeros(1,NSlots);

            offset = 0;
            SNR  = 10^(SNRdB/10);
            N0   = 1/sqrt(simLocal.NRxAnts*double(waveinfoLocal.Nfft)*SNR);
            nVar = N0^2*double(waveinfoLocal.Nfft);

            totalAntBitErrors = 0;
            totalAntBitsSent  = 0;

            for nslot = 0:NSlots-1

                carrier.NSlot = nslot;

                [pdschIndices,pdschIndicesInfo] = nrPDSCHIndices(carrier,pdsch);
                
                bitsPerSym  = Qm + gsm_idx_bits;
                G_target    = numel(pdschIndices) * bitsPerSym;
                G_scaled    = floor(G_target / Qm) * Qm;          % force valid rate-matching length
                trBlkSizes  = floor(G_scaled * pdschextra.TargetCodeRate / 8) * 8;

                for cwIdx = 1:pdsch.NumCodewords
                    if harqEntity.NewData(cwIdx)
                        trBlk = randi([0 1],trBlkSizes(cwIdx),1,'like',simLocal.Prototype);
                        setTransportBlock(encodeDLSCH,double(trBlk),cwIdx-1,harqEntity.HARQProcessID);
                        if harqEntity.SequenceTimeout(cwIdx)
                            resetSoftBuffer(decodeDLSCHLocal,cwIdx-1,harqEntity.HARQProcessID);
                        end
                    end
                end

                codedTrBlocks = encodeDLSCH(pdsch.Modulation,pdsch.NumLayers, ...
                    G_scaled,harqEntity.RedundancyVersion, ...
                    harqEntity.HARQProcessID,harqEntity.CBGTI);

                % pdschGrid = zeros(carrier.NSizeGrid*12, ...
                    % carrier.SymbolsPerSlot,simLocal.NTxAnts,"like",simLocal.Prototype);

                % ── Bit packing: Qm data bits + gsm_idx_bits combo-select bits, per RE ──               
                nBits  = numel(codedTrBlocks);
                nSym   = ceil(nBits / bitsPerSym);
                padLen = nSym*bitsPerSym - nBits;
                bitsPadded = [codedTrBlocks(:); zeros(padLen, 1, 'like', codedTrBlocks)];
                bitsPadded = reshape(bitsPadded, bitsPerSym, nSym).';

                modBits      = bitsPadded(:, 1:Qm);
                comboBitsMat = bitsPadded(:, Qm+1:end);
                comboIdx     = bi2de(comboBitsMat, 'left-msb') + 1;

                gsmSymbols = nrSymbolModulate(reshape(modBits.', [], 1), mod_scheme);

                assert(nSym <= numel(pdschIndices), ...
                    'GSM: nSym (%d) exceeds available REs (%d) at Qm=%d, gsm_idx_bits=%d', ...
                    nSym, numel(pdschIndices), Qm, gsm_idx_bits);

                reIdx = pdschIndices(1:nSym);
                [kSub, lSub] = ind2sub([carrier.NSizeGrid*12, carrier.SymbolsPerSlot], reIdx);

                MAX_ANT = size(this.raw_channel, 4);   % 32
                pdschGrid = zeros(carrier.NSizeGrid*12, carrier.SymbolsPerSlot, MAX_ANT, 'like', simLocal.Prototype);

                for iSym = 1:nSym
                    activeAnts = all_combos(comboIdx(iSym), :);
                    pdschGrid(kSub(iSym), lSub(iSym), activeAnts) = gsmSymbols(iSym) / sqrt(act_ants);
                end

                txWaveform1 = nrOFDMModulate(carrier, pdschGrid);   % already MAX_ANT-wide
                txWaveform  = txWaveform1;

                txWaveform = [txWaveform; zeros(maxChDelay, size(txWaveform,2))]; %#ok<AGROW>
                [rxWaveform,ofdmResponse,timingOffset] = channel(double(txWaveform),carrier);

                noise      = N0*randn(size(rxWaveform),"like",rxWaveform);
                rxWaveform = rxWaveform + noise;

                offset     = timingOffset;
                rxWaveform = rxWaveform(1+offset:end,:);

                rxGrid = nrOFDMDemodulate(carrier,rxWaveform);
                [K,L,R] = size(rxGrid);
                if (L < carrier.SymbolsPerSlot)
                    rxGrid = cat(2,rxGrid,zeros(K,carrier.SymbolsPerSlot-L,R));
                end

                % ──────── ML detection (Jeganathan metric) ──────────────────────
                H_full       = single(ofdmResponse(:,:,:,1:tx_ants));
                rxGridSingle = single(rxGrid);

                [kIdx, lIdx] = ind2sub([carrier.NSizeGrid*12, carrier.SymbolsPerSlot], reIdx);
                klIdx = sub2ind([carrier.NSizeGrid*12, carrier.SymbolsPerSlot], kIdx, lIdx);

                nRx_   = size(H_full, 3);
                Y      = reshape(rxGridSingle, [], nRx_);
                Y_ml   = Y(klIdx, :);
                H_flat = reshape(H_full, [], nRx_, tx_ants);
                H_ml   = H_flat(klIdx, :, :);

                estComboIdx   = zeros(nSym, 1);
                bestMetricVec = -Inf(nSym, 1, 'single');
                chunkSize     = 8192;

                Y3 = reshape(Y_ml, nSym, nRx_, 1);   % [nSym x Nr x 1] for broadcasting

                % ── Combo-selection matrix: B_full(ant,c)=1 if antenna 'ant' is used in combo c ──
                antIdxAll = all_combos(1:num_used_combos, :);          % [num_used_combos x act_ants]
                rowIdx    = antIdxAll(:);
                colIdx    = repmat((1:num_used_combos)', act_ants, 1);
                B_full    = zeros(tx_ants, num_used_combos, 'single');
                B_full(sub2ind(size(B_full), rowIdx, colIdx)) = 1;

                H_mat = reshape(H_ml, nSym*nRx_, tx_ants);             % view only, no copy cost

                for cStart = 1:chunkSize:num_used_combos
                    cEnd = min(cStart+chunkSize-1, num_used_combos);
                    nC   = cEnd - cStart + 1;

                    Bchunk  = B_full(:, cStart:cEnd);                        % [tx_ants x nC]
                    H_c_mat = (H_mat * Bchunk) / sqrt(single(act_ants));     % GEMM replaces gather+reshape+sum
                    H_c     = reshape(H_c_mat, nSym, nRx_, nC);

                    hHy  = sum(conj(H_c) .* Y3, 2);
                    num_ = abs(hHy).^2;
                    den_ = sum(real(conj(H_c).*H_c), 2);
                    metric = reshape(num_ ./ max(den_, single(1e-7)), nSym, nC);

                    [chunkBest, chunkIdx] = max(metric, [], 2);

                    updateMask = chunkBest > bestMetricVec;
                    bestMetricVec(updateMask) = chunkBest(updateMask);
                    estComboIdx(updateMask)   = cStart + chunkIdx(updateMask) - 1;
                end

                estBitsCombo = de2bi(estComboIdx - 1, gsm_idx_bits, 'left-msb');
                antBitErrors = sum(comboBitsMat(:) ~= estBitsCombo(:));
                totalAntBitErrors = totalAntBitErrors + antBitErrors;
                totalAntBitsSent  = totalAntBitsSent + numel(comboBitsMat);
                
                % ── Gather effective channel Hc per symbol using its estimated combo ──
                active_ants_per_sym = all_combos(estComboIdx, :);      % [nSym x act_ants]

                sIdx = repmat((1:nSym)', 1, nRx_, act_ants);
                rIdx = repmat(reshape(1:nRx_, 1, nRx_, 1), nSym, 1, act_ants);
                aIdx = repmat(reshape(active_ants_per_sym, nSym, 1, act_ants), 1, nRx_, 1);

                linIdx = sub2ind(size(H_ml), sIdx, rIdx, aIdx);
                Hc_all = sum(H_ml(linIdx), 3) / sqrt(single(act_ants));   % [nSym x nRx_]

                denom           = max(sum(real(conj(Hc_all).*Hc_all), 2), single(1e-7));   % [nSym x 1]
                effNoiseVar_all = nVar ./ denom;
                zEq_all         = sum(conj(Hc_all) .* Y_ml, 2) ./ denom;                    % [nSym x 1]
                
                % ── Demodulate once at unit noise variance (scalar, batched over all symbols) ──
                llrUnitVar = nrSymbolDemodulate(zEq_all, mod_scheme, single(1));   % [nSym*Qm x 1]
                
                % ── Rescale: approx-LLR magnitude is inversely proportional to noise variance,
                %    so LLR(sigma^2) = LLR(1) / sigma^2. This reproduces the original
                %    per-symbol nrSymbolDemodulate(zEq, mod, effNoiseVar) call exactly.
                llrUnitVar = reshape(llrUnitVar, Qm, nSym).';                      % [nSym x Qm]
                llrDataAll = llrUnitVar ./ effNoiseVar_all;                        % broadcast [nSym x 1] over Qm cols
                
                llrOut = zeros(nSym, bitsPerSym);
                llrOut(:, 1:Qm)     = llrDataAll;
                LLR_SAT             = 20;
                llrOut(:, Qm+1:end) = LLR_SAT * (1 - 2*estBitsCombo);

                dlschLLRs_flat = reshape(llrOut.', [], 1);
                dlschLLRs_flat = dlschLLRs_flat(1:nBits);
                dlschLLRs = {dlschLLRs_flat};

                decodeDLSCHLocal.TransportBlockLength = trBlkSizes;
                [decbits,blkerr,cbgerr] = decodeDLSCHLocal(dlschLLRs, ...
                    pdsch.Modulation,pdsch.NumLayers,harqEntity.RedundancyVersion, ...
                    harqEntity.HARQProcessID,harqEntity.CBGTI);

                simThroughput = simThroughput + sum(~blkerr .* trBlkSizes);
                maxThroughput = maxThroughput + sum(trBlkSizes);

                procstatus = updateAndAdvance(harqEntity,blkerr,trBlkSizes,G_scaled,cbgerr);

                harqvals(nslot+1) = ~blkerr;

            end

            totalCorrectAntBits = totalAntBitsSent - totalAntBitErrors;
            totalCorrectAntBits = max(0, totalCorrectAntBits); 
            totalFrameTimeSec = simLocal.NFrames * 10e-3;

            achiev_thr = 1e-6*simThroughput/(simLocal.NFrames*10e-3);
            achiev_thr_spatial = 1e-6 * (totalCorrectAntBits / totalFrameTimeSec);
            this.harq_ack_nack = mean(harqvals);
            frame_bler = 1 - this.harq_ack_nack;

            dataRow = [];
            dataRow = [achiev_thr, achiev_thr_spatial, this.SNR_dB, mcs_ind, tx_ants, act_ants];

            % ────────────── SNR random walk for next step ───────────────────
            delta_snr    = randi([-2,2],1);
            this.SNR_dB  = this.SNR_dB + delta_snr;
            this.SNR_dB  = max(-20, min(35, this.SNR_dB));
            snr_min      = -20;
            snr_max      = 35;
            this.normalized_snr = 2 * (this.SNR_dB - snr_min) / (snr_max - snr_min) - 1;

            % ────────────── Normalize for agent ───────────────────
            norm_prev_mcs   = this.prev_mcs_ind / 24;   
            norm_prev_alpha = this.prev_alpha / 4;

            % ────────────── Observation and Reward ───────────────────
            Observation = [this.normalized_snr; this.harq_ack_nack; norm_prev_mcs; norm_prev_alpha; this.csi_vector];
            max_thr     = 118.0;   % theoretical peak Mbps
            norm_thr    = achiev_thr / max_thr;      % [0, 1]
            
            max_spatial_capacity_Mbps = 1e-6 * numel(comboBitsMat) * NSlots / totalFrameTimeSec;
            if max_spatial_capacity_Mbps > 0
                norm_thr_sp = achiev_thr_spatial / max_spatial_capacity_Mbps;
            else
                norm_thr_sp = 0;
            end     

            if frame_bler == 1
                % The packet failed. Penalize the agent more heavily if it
                % greedily chose a high MCS, and less heavily if it chose a low MCS.
                Reward = -2.0 - (0.1 * mcs_ind);
            else
                % The packet succeeded. Reward it for high throughput.
                Reward = norm_thr + norm_thr_sp;
            end
            % fprintf('[R] norm_thr=%.4f | norm_thr_sp=%.4f | bler_penalty=%.4f | frame_bler=%.4f | R=%.4f\n', ...
            %     norm_thr, norm_thr_sp, bler_penalty_factor * frame_bler, frame_bler, Reward);

            % ────────────── Save current actions as previous indices for next step ───────────────────
            this.prev_mcs_ind = mcs_ind;
            this.prev_alpha   = alpha;

            % ────────────── Log the values in excel ───────────────────
            dataRow = [dataRow, this.harq_ack_nack, Reward];
            writematrix(dataRow, 'SM_env_july30.csv', 'WriteMode', 'append');

            IsDone = false;
        end

        function Observation = reset(this)
            [this.raw_channel, this.local_channel, ~] = SRS_Channel_gen(32);

            CORR_THRESHOLD = 0.75;
            this.combo_cache = cell(size(this.VALID_CONFIGS,1), 1);

            for cfgRow = 1:size(this.VALID_CONFIGS,1)
                tx_ants_c  = this.VALID_CONFIGS(cfgRow,1);
                act_ants_c = this.VALID_CONFIGS(cfgRow,2);

                estCh = this.raw_channel(:,:,:,1:tx_ants_c);
                [K,L,Nr,~] = size(estCh);

                all_combos_c = nchoosek(1:tx_ants_c, act_ants_c);
                num_combos_c = size(all_combos_c,1);
                target_c     = 2^floor(log2(num_combos_c));

                antEnergy_c   = squeeze(sum(mean(abs(estCh).^2, [1 2]), 3));
                comboEnergy_c = sum(antEnergy_c(all_combos_c), 2);

                Mp      = permute(reshape(estCh, K*L, Nr, tx_ants_c), [2 3 1]);
                G       = pagemtimes(pagectranspose(Mp), Mp);
                normAnt = sqrt(sum(abs(Mp).^2, 1));
                denom   = pagemtimes(pagectranspose(normAnt), normAnt);
                corrMat = mean(abs(G) ./ max(denom, eps), 3);

                pairPos = nchoosek(1:act_ants_c, 2);
                idx1    = all_combos_c(:, pairPos(:,1));
                idx2    = all_combos_c(:, pairPos(:,2));
                maxCorr = max(corrMat(sub2ind(size(corrMat), idx1, idx2)), [], 2);

                survivorIdx = find(maxCorr <= CORR_THRESHOLD);

                if numel(survivorIdx) >= target_c
                    [~, ord]  = sort(comboEnergy_c(survivorIdx), 'descend');
                    keepIdx   = survivorIdx(ord(1:target_c));
                else
                    shortfall    = target_c - numel(survivorIdx);
                    discardIdx   = setdiff((1:num_combos_c)', survivorIdx);
                    [~, ordDisc] = sort(comboEnergy_c(discardIdx), 'descend');
                    backfillIdx  = discardIdx(ordDisc(1:shortfall));
                    keepIdx      = [survivorIdx; backfillIdx];
                end

                this.combo_cache{cfgRow} = all_combos_c(keepIdx, :);
            end

            H_wb = squeeze(mean(this.raw_channel, [1 2]));   % 4 x 32
            frob_norm = norm(H_wb, 'fro');
            if frob_norm > 1e-8
                H_wb = H_wb / frob_norm; % Total matrix energy is now 1
            end
            H_wb_vect = reshape(H_wb,1,[]);
            H_wb_final = zeros(1,2*length(H_wb_vect));
            H_wb_final(1:2:length(H_wb_final)) = real(H_wb_vect);
            H_wb_final(2:2:length(H_wb_final)) = imag(H_wb_vect);
            this.csi_vector = H_wb_final';

            % Initialise tx_ants/act_ants to a random valid config
            % so that properties are never uninitialised before first step().
            cfg_idx      = randi(size(this.VALID_CONFIGS, 1));
            this.tx_ants  = this.VALID_CONFIGS(cfg_idx, 1);
            this.act_ants = this.VALID_CONFIGS(cfg_idx, 2);

            % Initialize default states for step 1 tracking context
            this.prev_mcs_ind = 0;
            this.prev_alpha   = cfg_idx - 1; % 0-based index

            % Normalize current states
            norm_prev_mcs   = this.prev_mcs_ind / 24;
            norm_prev_alpha = this.prev_alpha / 4;

            % Random starting SNR
            this.SNR_dB = randi([-20,35],1);
            snr_min     = -20;
            snr_max     = 35;
            this.normalized_snr = 2*(this.SNR_dB - snr_min)/(snr_max - snr_min) - 1;

            this.harq_ack_nack = 0;
            Observation = [this.normalized_snr; this.harq_ack_nack; norm_prev_mcs; norm_prev_alpha; this.csi_vector];
        end
    end
end

% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

