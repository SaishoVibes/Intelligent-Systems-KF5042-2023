function out = lstm_predict(in)
%#codegen

%   Copyright 2020 The MathWorks, Inc.

    persistent mynet;

    if isempty(mynet)
        mynet = coder.loadDeepLearningNetwork('textClassifierNetwork.mat');
    end

    out = predict(mynet, in);
end
