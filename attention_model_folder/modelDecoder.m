function [Y,context,hiddenState,attentionScores] = modelDecoder(parameters,X,context, ...
    hiddenState,Z,dropout)

% Embedding.
weights = parameters.emb.Weights;
X = embed(X,weights,DataFormat="CBT");

% RNN input.
sequenceLength = size(X,3);
Y = cat(1, X, repmat(context,[1 1 sequenceLength]));

% LSTM.
inputWeights = parameters.lstm.InputWeights;
recurrentWeights = parameters.lstm.RecurrentWeights;
bias = parameters.lstm.Bias;

initialCellState = dlarray(zeros(size(hiddenState)));

[Y,hiddenState] = lstm(Y,hiddenState,initialCellState, ...
    inputWeights,recurrentWeights,bias,DataFormat="CBT");

% Dropout.
mask = rand(size(Y),"like",Y) > dropout;
Y = Y.*mask;

% Attention.
weights = parameters.attention.Weights;
[context,attentionScores] = luongAttention(hiddenState,Z,weights);

% Concatenate.
Y = cat(1, Y, repmat(context,[1 1 sequenceLength]));

% Fully connect.
weights = parameters.fc.Weights;
bias = parameters.fc.Bias;
Y = fullyconnect(Y,weights,bias,DataFormat="CBT");

% Softmax.
Y = softmax(Y,DataFormat="CBT");

end