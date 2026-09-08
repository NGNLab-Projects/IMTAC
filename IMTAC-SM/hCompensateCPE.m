function pxschEq = hCompensateCPE(carrier,pxsch,pxschEq,rxGrid,estChannelGrid,noiseEst)
%   PXSCHEQ =
%   hCompensateCPE(CARRIER,PXSCH,PXSCHEQ,RXGRID,ESTCHANNELGRID,NOISEEST)
%   corrects the common phase error (CPE) based on PT-RS symbols for PDSCH
%   or PUSCH

%   Copyright 2025 The MathWorks, Inc.

    % Get reference grid and location map
    if isa(pxsch,"nrPDSCHConfig")
        reservedRE = {pxsch.ReservedRE};
    else
        reservedRE = {[]};
    end
    [refGridAllRS,locationMap] = nr5g.internal.evm.getI2ReferenceSignalGrid( ...
        carrier,pxsch,carrier.NSlot,reservedRE,class(rxGrid));
    dataLocation = 1;
    ptrsLocation = 3;

    % Get the equalized PXSCH and PT-RS symbols in the form of grid
    transformPrecoding = isa(pxsch,"nrPUSCHConfig") && pxsch.TransformPrecoding;
    eqGrid = nrResourceGrid(carrier,pxsch.NumLayers,OutputDataType=class(rxGrid));
    if transformPrecoding
        % When transform precoding is enabled, both PUSCH and PT-RS symbols
        % are part of the equalized data already. Map them to the grid
        % directly
        pxschIndices = find(locationMap == dataLocation | locationMap==ptrsLocation);
        eqGrid(pxschIndices) = pxschEq;
    else
        % When transform precoding is disabled, pxschEq contains equalized
        % PXSCH symbols only. Extract and equalize PT-RS symbols and then
        % combine them with the equalized PXSCH symbols

        % Extract and equalize PT-RS symbols
        locationMapPTRS = locationMap;
        locationMapPTRS(locationMap ~= ptrsLocation) = 0; % location map for PT-RS only
        eqGridPTRS = nr5g.internal.evm.equalizeChannel(rxGrid,estChannelGrid,locationMapPTRS,noiseEst);

        % Map PXSCH symbols to the grid
        pxschIndices = find(locationMap == dataLocation);
        eqGrid(pxschIndices) = pxschEq;
        % Combine PXSCH symbols and PT-RS symbols
        eqGrid = eqGrid+eqGridPTRS;
    end

    % Compensate common phase error (CPE)
    eqGrid = nr5g.internal.evm.compensateCPE(pxsch,eqGrid,refGridAllRS, ...
        locationMap,carrier.NSlot,carrier.CyclicPrefix);

    % Extract equalized PXSCH symbols and reshape to layers
    pxschEq = reshape(eqGrid(pxschIndices),[],pxsch.NumLayers);
end