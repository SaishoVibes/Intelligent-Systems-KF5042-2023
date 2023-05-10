function documents = transformText(str,startToken,stopToken)

str = strip(replace(str,""," "));
str = startToken + str + stopToken;
documents = tokenizedDocument(str,CustomTokens=[startToken stopToken]);

end