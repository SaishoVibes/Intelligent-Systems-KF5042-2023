function [context,attentionScores] = luongAttention(hiddenState,Z,weights)

numHeads = 1;
queries = hiddenState;
keys = pagemtimes(weights,Z);
values = Z;

[context,attentionScores] = attention(queries,keys,values,numHeads, ...
    Scale=1, ...
    DataFormat="CBT");

end