function documents = preprocessText_2(textData)
% Tokenize the text
% 刪除StopWords, punctuation，並做normalize

% ref:
% https://www.mathworks.com/help/textanalytics/ug/classify-text-data-using-deep-learning.html
% https://www.mathworks.com/help/textanalytics/ug/create-simple-text-model-for-classification.html

% Tokenize the text.
documents = tokenizedDocument(textData);

% Remove a list of stop words then lemmatize the words. To improve
% lemmatization, first use addPartOfSpeechDetails.
documents = addPartOfSpeechDetails(documents);
documents = removeStopWords(documents);
documents = normalizeWords(documents,'Style','lemma');

% Erase punctuation.
documents = erasePunctuation(documents);

end