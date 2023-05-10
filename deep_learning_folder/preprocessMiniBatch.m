function [XSource,XTarget,mask,decoderInput] = preprocessMiniBatch(dataSource,dataTarget,encGerman,encEnglish)

documentsGerman = cat(1,dataSource{:});
XSource = preprocessPredictors(documentsGerman,encGerman);

documentsEngligh = cat(1,dataTarget{:});
sequencesTarget = doc2sequence(encEnglish,documentsEngligh,PaddingDirection="none");

[XTarget,mask] = padsequences(sequencesTarget,2,PaddingValue=1);

decoderInput = XTarget(:,1,:);
XTarget(:,1,:) = [];
mask(:,1,:) = [];

end