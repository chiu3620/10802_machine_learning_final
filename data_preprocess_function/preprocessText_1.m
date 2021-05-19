function documents = preprocessText_1(textData)
% Tokenize the text
% 刪除StopWords, punctuation，並做normalize
% 刪除太短跟太長的句子

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

% Remove words with 2 or fewer characters, and words with 15 or more
% characters.
documents = removeShortWords(documents,2);
documents = removeLongWords(documents,15);

end