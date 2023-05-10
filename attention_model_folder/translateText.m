function strTranslated = translateText(parameters,strSource,maxSequenceLength,miniBatchSize, ...
    encSource,encTarget,startToken,stopToken,delimiter)

% Transform text.
documentsSource = transformText(strSource,startToken,stopToken);
sequencesSource = doc2sequence(encSource,documentsSource, ...
    PaddingDirection="right", ...
    PaddingValue=encSource.NumWords + 1);

% Convert to dlarray.
X = cat(3,sequencesSource{:});
X = permute(X,[1 3 2]);
X = dlarray(X);

% Initialize output.
numObservations = numel(strSource);
strTranslated = strings(numObservations,1);

% Loop over mini-batches.
numIterations = ceil(numObservations / miniBatchSize);
for i = 1:numIterations
    idxMiniBatch = (i-1)*miniBatchSize+1:min(i*miniBatchSize,numObservations);
    miniBatchSize = numel(idxMiniBatch);

    % Encode using model encoder.
    sequenceLengths = [];
    [Z, hiddenState] = modelEncoder(parameters.encoder,X(:,idxMiniBatch,:),sequenceLengths);

    % Decoder predictions.
    doTeacherForcing = false;
    dropout = 0;
    decoderInput = repmat(word2ind(encTarget,startToken),[1 miniBatchSize]);
    decoderInput = dlarray(decoderInput);
    Y = decoderPredictions(parameters.decoder,Z,decoderInput,hiddenState,dropout, ...
        doTeacherForcing,maxSequenceLength);
    [~, idxPred] = max(extractdata(Y),[],1);

    % Keep translating flag.
    idxStop = word2ind(encTarget,stopToken);
    keepTranslating = idxPred ~= idxStop;

    % Loop over time steps.
    t = 1;
    while t <= maxSequenceLength && any(keepTranslating(:,:,t))

        % Update output.
        newWords = ind2word(encTarget, idxPred(:,:,t))';
        idxUpdate = idxMiniBatch(keepTranslating(:,:,t));
        strTranslated(idxUpdate) = strTranslated(idxUpdate) + delimiter + newWords(keepTranslating(:,:,t));

        t = t + 1;
    end
end

end