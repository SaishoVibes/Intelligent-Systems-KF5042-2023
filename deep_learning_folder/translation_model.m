% https://www.mathworks.com/help/textanalytics/ug/language-translation-using-deep-learning-example.html#
clc % clears screen
clear % clears workspace
discardProp = 0.70; % discards 70% of population to consolidate data
%% Load Training Data
% Download the dataset and make a table
downloadFolder = tempdir;
url = "https://www.manythings.org/anki/deu-eng.zip";
filename = fullfile(downloadFolder,"deu-eng.zip");
dataFolder = fullfile(downloadFolder,"deu-eng");

% Downloads and initiates zip/folder
if ~exist(dataFolder,"dir")
    fprintf("Downloading English-German Tab-delimited Bilingual Sentence Pairs data set (7.6 MB)... ")
    websave(filename,url);
    unzip(filename,dataFolder);
    fprintf("Done.\n")
end

filename = fullfile(dataFolder,"deu.txt");

%Sets table to show first 8 options
opts = delimitedTextImportOptions(...
    Delimiter="\t", ...
    VariableNames=["Target" "Source" "License"], ...
    SelectedVariableNames=["Source" "Target"], ...
    VariableTypes=["string" "string" "string"], ...
    Encoding="UTF-8");
data = readtable(filename, opts);
head(data)
idx = size(data,1) - floor(discardProp*size(data,1)) + 1;
data(idx:end,:) = [];
size(data,1)
% Prepare training
trainingProp = 0.9;
idx = randperm(size(data,1),floor(trainingProp*size(data,1)));
dataTrain = data(idx,:);
dataTest = data;
dataTest(idx,:) = [];
head(dataTrain) % view table of training data
numObservationsTrain = size(dataTrain,1) % View the number of training observations.

%% Preprocess Data
documentsGerman = preprocessText(dataTrain.Source); % Tokens the text and splits it into words
encGerman = wordEncoding(documentsGerman); % Create a wordEncoding object that maps tokens to a numeric index and vice versa using a vocabulary.

% Convert the target data to sequences using the same steps.
documentsEnglish = preprocessText(dataTrain.Target);
encEnglish = wordEncoding(documentsEnglish);

% View the vocabulary sizes of the source and target encodings.
numWordsGerman = encGerman.NumWords
numWordsEnglish = encEnglish.NumWords

%% Define Encoder and Decoder Networks (will do at home)

embeddingDimension = 128; %dimension and units of embed
numHiddenUnits = 128;
%layers of the encoder and definition
[lgraphEncoder,lgraphDecoder] = languageTranslationLayers(embeddingDimension,numHiddenUnits,numWordsGerman,numWordsEnglish);

%convert to dlnetwork for training loop
netEncoder = dlnetwork(lgraphEncoder);
netDecoder = dlnetwork(lgraphDecoder);

%output
netDecoder.OutputNames = ["softmax" "context" "lstm2/hidden" "lstm2/cell"];

%% specify training options
%time and learn rate
miniBatchSize = 64;
numEpochs = 3; % moved down from 15 due to time needed
learnRate = 0.005;
%adam optimization initialize
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;
%set epsilon values to go down over time
epsilonStart = 0.5;
epsilonEnd = 0;
%Sort the training sequences by sequence length.
sequenceLengths = doclength(documentsGerman);
[~,idx] = sort(sequenceLengths);
documentsGerman = documentsGerman(idx);
documentsEnglish = documentsEnglish(idx);
%% train model
%arrays to store and combine the data
adsSource = arrayDatastore(documentsGerman);
adsTarget = arrayDatastore(documentsEnglish);
cds = combine(adsSource,adsTarget);
%automatically train mini batches and discard partial ones
mbq = minibatchqueue(cds,4, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@(X,Y) preprocessMiniBatch(X,Y,encGerman,encEnglish), ...
    MiniBatchFormat=["CTB" "CTB" "CTB" "CTB"], ...
    PartialMiniBatch="discard");
% initialize the process plot
figure
C = colororder;
lineLossTrain = animatedline(Color=C(2,:));

xlabel("Iteration")
ylabel("Loss")
ylim([0 inf])
grid on
%For the encoder and decoder networks, initialize the values for Adam optimization.
trailingAvgEncoder = [];
trailingAvgSqEncoder = [];
trailingAvgDecder = [];
trailingAvgSqDecoder = [];
%Create an array of ϵ values for scheduled sampling.
numIterationsPerEpoch = floor(numObservationsTrain/miniBatchSize);
%numIterationsPerEpoch = 128;
numIterations = numIterationsPerEpoch * numEpochs;
epsilon = linspace(epsilonStart,epsilonEnd,numIterations);
%train model for each iteration
iteration = 0;
start = tic;
lossMin = inf;
reset(mbq)

% Loop over epochs.
for epoch = 1:numEpochs

    % Loop over mini-batches.
    while hasdata(mbq)
        iteration = iteration + 1;

        % Read mini-batch of data.
        [X,T,maskT,decoderInput] = next(mbq);
        
        % Compute loss and gradients.
        [loss,gradientsEncoder,gradientsDecoder,YPred] = dlfeval(@modelLoss,netEncoder,netDecoder,X,T,maskT,decoderInput,epsilon(iteration));
        
        % Update network learnable parameters using adamupdate.
        [netEncoder, trailingAvgEncoder, trailingAvgSqEncoder] = adamupdate(netEncoder,gradientsEncoder,trailingAvgEncoder,trailingAvgSqEncoder, ...
            iteration,learnRate,gradientDecayFactor,squaredGradientDecayFactor);

        [netDecoder, trailingAvgDecder, trailingAvgSqDecoder] = adamupdate(netDecoder,gradientsDecoder,trailingAvgDecder,trailingAvgSqDecoder, ...
            iteration,learnRate,gradientDecayFactor,squaredGradientDecayFactor);

        % Generate translation for plot.
        if iteration == 1 || mod(iteration,10) == 0
            strGerman = ind2str(X(:,1,:),encGerman);
            strEnglish = ind2str(T(:,1,:),encEnglish,Mask=maskT);
            strTranslated = ind2str(YPred(:,1,:),encEnglish);
        end

        % Display training progress.
        D = duration(0,0,toc(start),Format="hh:mm:ss");
        loss = double(gather(extractdata(loss)));
        addpoints(lineLossTrain,iteration,loss)
        title( ...
            "Epoch: " + epoch + ", Elapsed: " + string(D) + newline + ...
            "Source: " + strGerman + newline + ...
            "Target: " + strEnglish + newline + ...
            "Training Translation: " + strTranslated)

        drawnow
        
        % Save best network.
        if loss < lossMin
            lossMin = loss;
            netBest.netEncoder = netEncoder;
            netBest.netDecoder = netDecoder;
            netBest.loss = loss;
            netBest.iteration = iteration;
            netBest.D = D;
        end
    end

    % Shuffle.
    shuffle(mbq);
end

%Add the word encodings to the netBest structure and save the structure in a MAT file.
netBest.encGerman = encGerman;
netBest.encEnglish = encEnglish;

D = datetime("now",Format="yyyy_MM_dd__HH_mm_ss");
filename = "net_best__" + string(D) + ".mat";
save(filename,"netBest");

%Extract the best network from netBest.
netEncoder = netBest.netEncoder;
netDecoder = netBest.netDecoder;



%% testing the model (run section only after making model)
fprintf("1") %adding to see progress
%Translate the test data using the translateText function.
strTranslatedTest = translateText(netEncoder,netDecoder,encGerman,encEnglish,dataTest.Source);

%View a random selection of the test source text, target text, and predicted translations in a table.
fprintf("2") %adding to see progress
numObservationsTest = size(dataTest,1);
idx = randperm(numObservationsTest,8);
tbl = table;
tbl.Source = dataTest.Source(idx);
tbl.Target = dataTest.Target(idx);
tbl.Translated = strTranslatedTest(idx)

fprintf("3") %adding to see progress
%Specify empty start and stop tokens, as these are not used in the translation.
candidates = preprocessText(strTranslatedTest,StartToken="",StopToken="");
references = preprocessText(dataTest.Target,StartToken="",StopToken="");

fprintf("4") %adding to see progress
%Determine the length of the shortest candidate document.
minLength = min([doclength(candidates); doclength(references)])

fprintf("5") %adding to see progress
%manually set weight if length is larger
if minLength < 4
    ngramWeights = ones(1,minLength) / minLength;
else
    ngramWeights = [0.25 0.25 0.25 0.25];
end

fprintf("6") %adding to see progress
%Calculate the BLEU evaluation scores
for i = 1:numObservationsTest
    score(i) = bleuEvaluationScore(candidates(i),references(i),NgramWeights=ngramWeights);
end

fprintf("7") %adding to see progress
%Visualize the BLEU evaluation scores in a histogram.
figure
histogram(score);
title("BLEU Evaluation Scores")
xlabel("Score")
ylabel("Frequency")

fprintf("8") %adding to see progress
%View a table of some of the best translations.
[~,idxSorted] = sort(score,"descend");
idx = idxSorted(1:8);
tbl = table;
tbl.Source = dataTest.Source(idx);
tbl.Target = dataTest.Target(idx);
tbl.Translated = strTranslatedTest(idx)

fprintf("9") %adding to see progress
%View a table of some of the worst translations.
idx = idxSorted(end-7:end);
tbl = table;
tbl.Source = dataTest.Source(idx);
tbl.Target = dataTest.Target(idx);
tbl.Translated = strTranslatedTest(idx)

fprintf("10") %adding to see progress
%Generate Translations
strGermanNew = [
    "Wie geht es Dir heute?"
    "Wie heißen Sie?"
    "Das Wetter ist heute gut."];
%Translate the text using the translateText, function listed at the end of the example.
strTranslatedNew = translateText(netEncoder,netDecoder,encGerman,encEnglish,strGermanNew)
