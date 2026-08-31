function [raw_channel, channel, maxChDelay] = SRS_Channel_gen(tx_ants)    
    simParameters = struct();       % Clear simParameters variable to contain all key simulation parameters
    simParameters.NFrames = 1;      % Number of 10 ms frames
    simParameters.NTxAnts = tx_ants;
    simParameters.NRxAnts = 4;
    simParameters.DataType = 'single';
    simParameters.Prototype =  zeros(0,0,simParameters.DataType);
    
    simParameters.Carrier = nrCarrierConfig;
    simParameters.Carrier.NSizeGrid = 51;
    simParameters.Carrier.SubcarrierSpacing = 30;
    simParameters.Carrier.CyclicPrefix = 'Normal';
    simParameters.Carrier.NCellID = 1;
    waveformInfo = nrOFDMInfo(simParameters.Carrier);
    
    simParameters.DelayProfile = 'CDL-C';
    simParameters.DelaySpread = 300e-9;
    simParameters.MaximumDopplerShift = 0;
    
    channel = nrCDLChannel;
    channel = hArrayGeometry(channel,simParameters.NTxAnts,simParameters.NRxAnts);
    simParameters.NTxAnts = prod(channel.TransmitAntennaArray.Size);
    simParameters.NRxAnts = prod(channel.ReceiveAntennaArray.Size);
    channel.DelayProfile = simParameters.DelayProfile;
    channel.DelaySpread = simParameters.DelaySpread;
    channel.MaximumDopplerShift = simParameters.MaximumDopplerShift;
    channel.SampleRate = waveformInfo.SampleRate;
    channel.ChannelResponseOutput = 'ofdm-response';
    channel.CarrierFrequency = 3.5e9;
    channel.Seed = randi(1e6);
    chInfo = info(channel);
    maxChDelay = chInfo.MaximumChannelDelay;
    
    % Raw_channel has the complete 612 x 14 x 4 x 32 channel matrix
    raw_channel = getInitialChannelEstimate(simParameters.Carrier,channel,simParameters.DataType,maxChDelay);

end


function estChannelGrid = getInitialChannelEstimate(carrier,propchannel,dataType,maxChDelay)
    % Obtain channel estimate before first transmission.
    ofdmInfo = nrOFDMInfo(carrier);
    % Clone of the channel
    chClone = propchannel.clone();
    chClone.release();
    % No filtering needed to get perfect channel estimate
    chClone.ChannelFiltering = false;
    chClone.OutputDataType = dataType;
    chClone.NumTimeSamples = (ofdmInfo.SampleRate/1000/carrier.SlotsPerSubframe)+maxChDelay;
    % Get the perfect channel estimate
    estChannelGrid = chClone(carrier);
end

function cdl = hArrayGeometry(cdl,NTxAnts,NRxAnts,varargin)
    if (nargin==3)
        linkDirection = 'downlink';
    else
        linkDirection = varargin{1};
    end
    if (strcmpi(linkDirection,'downlink'))
        txArray = bsArrayGeometry(cdl.TransmitAntennaArray,NTxAnts);
        rxArray = ueArrayGeometry(cdl.ReceiveAntennaArray,NRxAnts);
    else % uplink
        txArray = ueArrayGeometry(cdl.TransmitAntennaArray,NTxAnts);
        rxArray = bsArrayGeometry(cdl.ReceiveAntennaArray,NRxAnts);
    end
    % Update CDL channel arrays configuration
    cdl.TransmitAntennaArray = txArray;
    cdl.ReceiveAntennaArray = rxArray;
end

function array = bsArrayGeometry(array,nBsAnts)
    % Setup the base station antenna geometry
    % Table of antenna panel array configurations
    % M:  no. of rows in each antenna panel
    % N:  no. of columns in each antenna panel
    % P:  no. of polarizations (1 or 2)
    % Mg: no. of rows in the array of panels
    % Ng: no. of columns in the array of panels
    % Row format: [M  N   P   Mg  Ng]
    antArraySizes = ...
        [1   1   1   1   1;   % 1 ants
        1   1   2   1   1;   % 2 ants
        2   1   2   1   1;   % 4 ants
        2   2   2   1   1;   % 8 ants
        2   4   2   1   1;   % 16 ants
        4   4   2   1   1;   % 32 ants
        4   4   2   1   2;   % 64 ants
        4   8   2   1   2;   % 128 ants
        4   8   2   2   2;   % 256 ants
        8   8   2   2   2;   % 512 ants
        8  16   2   2   2];  % 1024 ants
    antselected = min(1+ceil(log2(nBsAnts)),size(antArraySizes,1));
    array.Size = antArraySizes(antselected,:);
    % Adjust element spacing to avoid panel overlaps
    array.ElementSpacing(3) = array.Size(1)*array.ElementSpacing(1);
    array.ElementSpacing(4) = array.Size(2)*array.ElementSpacing(2);
end

function array = ueArrayGeometry(array,nUeAnts)
    % Setup the UE antenna geometry
    if nUeAnts == 1
        % In the following settings, the number of rows in antenna array, 
        % columns in antenna array, polarizations, row array panels and the
        % columns array panels are all 1
        arraySize = ones(1,5);
    else
        % In the following settings, the no. of rows in antenna array is
        % nUeAnts/2, the no. of columns in antenna array is 1, the no.
        % of polarizations is 2, the no. of row array panels is 1 and the
        % no. of column array panels is 1. The values can be changed to
        % create alternative antenna setups
        arraySize = [ceil(nUeAnts/2),1,2,1,1];
    end
    array.Size = arraySize; 
end