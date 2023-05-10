function XSource = preprocessPredictors(documentsGerman,encGerman)

sequencesSource = doc2sequence(encGerman,documentsGerman,PaddingDirection="none");
XSource = padsequences(sequencesSource,2);

end