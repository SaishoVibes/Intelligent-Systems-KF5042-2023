%download
%filename = fullfile("romanNumerals.csv");
filename = fullfile("deu.csv"); %filename changed to be the dataset converted into a csv file.

options = detectImportOptions(filename, ...
    TextType="string", ...
    ReadVariableNames=false);
options.VariableNames = ["Source" "Target"];
options.VariableTypes = ["string" "string"];

data = readtable(filename,options);

%Split the data into training and test partitions containing 50% of the data each.

idx = randperm(size(data,1),500);
dataTrain = data(idx,:);
dataTest = data;
dataTest(idx,:) = [];

%View some of the decimal-Roman numeral pairs.
head(dataTrain)

%preprocess data
startToken = "<start>";
stopToken = "<stop>";

strSource = dataTrain.Source;
documentsSource = transformText(strSource,startToken,stopToken);

%wordencoding obj
encSource = wordEncoding(documentsSource);

%convert to sequence
sequencesSource = doc2sequence(encSource,documentsSource,PaddingDirection="none");

%Convert the target data to sequences using the same steps.
strTarget = dataTrain.Target;
documentsTarget = transformText(strTarget,startToken,stopToken);
encTarget = wordEncoding(documentsTarget);
sequencesTarget = doc2sequence(encTarget,documentsTarget,PaddingDirection="none");

%sort by length to ensure similar training times
sequenceLengths = cellfun(@(sequence) size(sequence,2),sequencesSource);
[~,idx] = sort(sequenceLengths);
sequencesSource = sequencesSource(idx);
sequencesTarget = sequencesTarget(idx);

%Create arrayDatastore objects containing the source and target data and combine them using the combine function.
sequencesSourceDs = arrayDatastore(sequencesSource,OutputType="same");
sequencesTargetDs = arrayDatastore(sequencesTarget,OutputType="same");

sequencesDs = combine(sequencesSourceDs,sequencesTargetDs);

%model parameter init
embeddingDimension = 128;
numHiddenUnits = 100;
dropout = 0.05;

%encoder model parameter init
inputSize = encSource.NumWords + 1;
sz = [embeddingDimension inputSize];
mu = 0;
sigma = 0.01;
parameters.encoder.emb.Weights = initializeGaussian(sz,mu,sigma);

%Initialize the learnable parameters for the encoder LSTM operation.
sz = [4*numHiddenUnits embeddingDimension];
numOut = 4*numHiddenUnits;
numIn = embeddingDimension;

parameters.encoder.lstm.InputWeights = initializeGlorot(sz,numOut,numIn);
parameters.encoder.lstm.RecurrentWeights = initializeOrthogonal([4*numHiddenUnits numHiddenUnits]);
parameters.encoder.lstm.Bias = initializeUnitForgetGate(numHiddenUnits);

%init decoder model param
outputSize = encTarget.NumWords + 1;
sz = [embeddingDimension outputSize];
mu = 0;
sigma = 0.01;
parameters.decoder.emb.Weights = initializeGaussian(sz,mu,sigma);
%Initialize the weights of the attention mechanism using the Glorot initializer 
%using the initializeGlorot function.
sz = [numHiddenUnits numHiddenUnits];
numOut = numHiddenUnits;
numIn = numHiddenUnits;
parameters.decoder.attention.Weights = initializeGlorot(sz,numOut,numIn);

%Initialize the learnable parameters for the decoder LSTM operation.
sz = [4*numHiddenUnits embeddingDimension+numHiddenUnits];
numOut = 4*numHiddenUnits;
numIn = embeddingDimension + numHiddenUnits;

parameters.decoder.lstm.InputWeights = initializeGlorot(sz,numOut,numIn);
parameters.decoder.lstm.RecurrentWeights = initializeOrthogonal([4*numHiddenUnits numHiddenUnits]);
parameters.decoder.lstm.Bias = initializeUnitForgetGate(numHiddenUnits);

%Initialize the learnable parameters for the decoder fully connected operation:
sz = [outputSize 2*numHiddenUnits];
numOut = outputSize;
numIn = 2*numHiddenUnits;

parameters.decoder.fc.Weights = initializeGlorot(sz,numOut,numIn);
parameters.decoder.fc.Bias = initializeZeros([outputSize 1]);

%Train with a mini-batch size of 32 for 100 epochs with a learning rate of 0.001.
miniBatchSize = 32;
numEpochs = 100;
learnRate = 0.001;

%Initialize the options from Adam.
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;

%% train model

%mini batch output
numMiniBatchOutputs = 4;

mbq = minibatchqueue(sequencesDs,numMiniBatchOutputs,...
    MiniBatchSize=miniBatchSize,...
    MiniBatchFcn=@(x,t) preprocessMiniBatch(x,t,inputSize,outputSize));

%Initialize the values for the adamupdate function.
trailingAvg = [];
trailingAvgSq = [];

%Calculate the total number of iterations for the training progress monitor
numObservationsTrain = numel(sequencesSource);
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;

%Initialize the training progress monitor. 
monitor = trainingProgressMonitor( ...
    Metrics="Loss", ...
    Info="Epoch", ...
    XLabel="Iteration");

%training loop for each mini batch
epoch = 0;
iteration = 0;

% Loop over epochs.
while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;

    reset(mbq);

    % Loop over mini-batches.
    while hasdata(mbq) && ~monitor.Stop
        iteration = iteration + 1;

        [X,T,sequenceLengthsSource,maskSequenceTarget] = next(mbq);

        % Compute loss and gradients.
        [loss,gradients] = dlfeval(@modelLoss,parameters,X,T,sequenceLengthsSource,...
            maskSequenceTarget,dropout);

        % Update parameters using adamupdate.
        [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients,trailingAvg,trailingAvgSq,...
            iteration,learnRate,gradientDecayFactor,squaredGradientDecayFactor);

        % Normalize loss by sequence length.
        loss = loss ./ size(T,3);

        % Update the training progress monitor. 
        recordMetrics(monitor,iteration,Loss=loss);
        updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
        monitor.Progress = 100*iteration/numIterations;
    end
end

%% generate translations

%Preprocess the text data using the same steps as when training. 
strSource = dataTest.Source;
strTarget = dataTest.Target;

%Translate the text using the modelPredictions function.
maxSequenceLength = 10;
delimiter = "";

strTranslated = translateText(parameters,strSource,maxSequenceLength,miniBatchSize, ...
    encSource,encTarget,startToken,stopToken,delimiter);

%Create a table containing the test source text, target text, and translations.
tbl = table;
tbl.Source = strSource;
tbl.Target = strTarget;
tbl.Translated = strTranslated;

%View a random selection of the translations.
idx = randperm(size(dataTest,1),miniBatchSize);
tbl(idx,:)

