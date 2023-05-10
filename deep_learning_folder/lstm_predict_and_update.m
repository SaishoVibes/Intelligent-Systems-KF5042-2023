function out = lstm_predict_and_update(in)
%#codegen

%   Copyright 2020 The MathWorks, Inc.

    persistent mynet;

    if isempty(mynet)
        mynet = coder.loadDeepLearningNetwork('textClassifierNetwork.mat');
    end

    [mynet, out] = predictAndUpdateState(mynet,in);
end
