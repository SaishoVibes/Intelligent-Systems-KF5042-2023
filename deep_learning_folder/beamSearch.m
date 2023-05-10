function str = beamSearch(X,netEncoder,netDecoder,encEnglish,args)

% Parse input arguments.
arguments
    X
    netEncoder
    netDecoder
    encEnglish
    
    args.BeamIndex = 3;
    args.MaxNumWords = 10;
end

beamIndex = args.BeamIndex;
maxNumWords = args.MaxNumWords;
startToken = "<start>";
stopToken = "<stop>";

% Encoder predictions.
[Z, hiddenState, cellState] = predict(netEncoder,X);

% Initialize context.
miniBatchSize = size(X,2);
numHiddenUnits = size(Z,1);
context = zeros([numHiddenUnits miniBatchSize],"like",Z);
context = dlarray(context,"CB");

% Initialize candidates.
candidates = struct;
candidates.Words = startToken;
candidates.Score = 0;
candidates.StopFlag = false;
candidates.HiddenState = hiddenState;
candidates.CellState = cellState;

% Loop over words.
t = 0;
while t < maxNumWords
    t = t + 1;

    candidatesNew = [];

    % Loop over candidates.
    for i = 1:numel(candidates)

        % Stop generating when stop token is predicted.
        if candidates(i).StopFlag
            continue
        end

        % Candidate details.
        words = candidates(i).Words;
        score = candidates(i).Score;
        hiddenState = candidates(i).HiddenState;
        cellState = candidates(i).CellState;

        % Predict next token.
        decoderInput = word2ind(encEnglish,words(end));
        decoderInput = dlarray(decoderInput,"CBT");

        [YPred,context,hiddenState,cellState] = predict(netDecoder,decoderInput,hiddenState,cellState,context,Z, ...
            Outputs=["softmax" "context" "lstm2/hidden" "lstm2/cell"]);

        % Find top predictions.
        [scoresTop,idxTop] = maxk(extractdata(YPred),beamIndex);
        idxTop = gather(idxTop);

        % Loop over top predictions.
        for j = 1:beamIndex
            candidate = struct;

            % Determine candidate word and score.
            candidateWord = ind2word(encEnglish,idxTop(j));
            candidateScore = scoresTop(j);

            % Set stop translating flag.
            if candidateWord == stopToken
                candidate.StopFlag = true;
            else
                candidate.StopFlag = false;
            end

            % Update candidate details.
            candidate.Words = [words candidateWord];
            candidate.Score = score + log(candidateScore);
            candidate.HiddenState = hiddenState;
            candidate.CellState = cellState;

            % Add to new candidates.
            candidatesNew = [candidatesNew candidate];
        end
    end

    % Get top candidates.
    [~,idx] = maxk([candidatesNew.Score],beamIndex);
    candidates = candidatesNew(idx);

    % Stop predicting when all candidates have stop token.
    if all([candidates.StopFlag])
        break
    end
end

% Get top candidate.
words = candidates(1).Words;

% Convert to string scalar.
words(ismember(words,[startToken stopToken])) = [];
str = join(words);

end