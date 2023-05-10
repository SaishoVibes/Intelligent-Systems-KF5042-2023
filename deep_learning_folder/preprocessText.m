function documents = preprocessText(str,args)

arguments
    str
    args.StartToken = "<start>";
    args.StopToken = "<stop>";
end

startToken = args.StartToken;
stopToken = args.StopToken;

str = lower(str);
str = startToken + str + stopToken;
documents = tokenizedDocument(str,CustomTokens=[startToken stopToken]);

end
%function documents = preprocessText(textData)

% Tokenize the text.
%documents = tokenizedDocument(textData);

% Convert to lowercase.
%documents = lower(documents);

% Erase punctuation.
%documents = erasePunctuation(documents);

%end