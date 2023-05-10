%% model loss
function [loss,gradientsE,gradientsD,YPred] = modelLoss(netEncoder,netDecoder,X,T,maskT,decoderInput,epsilon)

% Forward through encoder.
[Z, hiddenState, cellState] = forward(netEncoder,X);

% Decoder output.
Y = decoderPredictions(netDecoder,Z,T,hiddenState,cellState,decoderInput,epsilon);

% Sparse cross-entropy loss.
loss = sparseCrossEntropy(Y,T,maskT);

% Update gradients.
[gradientsE,gradientsD] = dlgradient(loss,netEncoder.Learnables,netDecoder.Learnables);

% For plotting, return loss normalized by sequence length.
sequenceLength = size(T,3);
loss = loss ./ sequenceLength;

% For plotting example translations, return the decoder output.
YPred = onehotdecode(Y,1:size(Y,1),1,"single");

end
