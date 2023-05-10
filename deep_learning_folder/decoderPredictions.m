function Y = decoderPredictions(netDecoder,Z,T,hiddenState,cellState,decoderInput,epsilon)

% Initialize context.
numHiddenUnits = size(Z,1);
miniBatchSize = size(Z,2);
context = zeros([numHiddenUnits miniBatchSize],"like",Z);
context = dlarray(context,"CB");

% Initialize output.
idx = (netDecoder.Learnables.Layer == "fc" & netDecoder.Learnables.Parameter=="Bias");
numClasses = numel(netDecoder.Learnables.Value{idx});
sequenceLength = size(T,3);
Y = zeros([numClasses miniBatchSize sequenceLength],"like",Z);
Y = dlarray(Y,"CBT");

% Forward start token through decoder.
[Y(:,:,1),context,hiddenState,cellState] = forward(netDecoder,decoderInput,hiddenState,cellState,context,Z);

% Loop over remaining time steps.
for t = 2:sequenceLength

    % Scheduled sampling. Randomly select previous target or previous
    % prediction.
    if rand < epsilon
        % Use target value.
        decoderInput = T(:,:,t-1);
    else
        % Use previous prediction.
        [~,Yhat] = max(Y(:,:,t-1),[],1);
        decoderInput = Yhat;
    end

    % Forward through decoder.
    [Y(:,:,t),context,hiddenState,cellState] = forward(netDecoder,decoderInput,hiddenState,cellState,context,Z);
end

end