import numpy as np

class NaiveBayes:
    def __init__(self):
        self.class_priors = {}  # Stores log P(C)
        self.word_counts = {}  # Stores word frequencies per class
        self.class_word_totals = {}  # Total words per class
        self.vocab = set()  # Vocabulary set
        self.smoothening = 1.0
        
    def fit(self, df, smoothening, class_col="Class Index", text_col="Tokenized Description"):
        """Learn the parameters of the model from the training data.
        Classes are 1-indexed.

        Args:
            df (pd.DataFrame): The training data containing columns class_col and text_col.
                Each entry of text_col is a list of tokens.
            smoothening (float): The Laplace smoothening parameter.
        """
        self.smoothening = smoothening
        class_counts = df[class_col].value_counts().to_dict()
        total_docs = len(df)
        
        # Compute prior probabilities P(C)
        self.class_priors = {c: np.log(count / total_docs) for c, count in class_counts.items()}
        
        # Initialize structures for word counts
        self.word_counts = {cls: {} for cls in class_counts}
        self.class_word_totals = {cls: 0 for cls in class_counts}
        
        # Compute word counts per class
        for _, row in df.iterrows():
            cls = row[class_col]
            tokens = row[text_col]
            self.class_word_totals[cls] += len(tokens)
            
            for token in tokens:
                if token not in self.word_counts[cls]:
                    self.word_counts[cls][token] = 0
                self.word_counts[cls][token] += 1
                self.vocab.add(token)
    
    def predict(self, df, text_col="Tokenized Description", predicted_col="Predicted"):
        """
        Predict the class of the input data by filling up column predicted_col in the input dataframe.

        Args:
            df (pd.DataFrame): The testing data containing column text_col.
                Each entry of text_col is a list of tokens.
        """
        predictions = []
        V = len(self.vocab)  # Vocabulary size
        
        for _, row in df.iterrows():
            tokens = row[text_col]
            class_scores = {}
            
            # Compute log-probabilities for each class
            for cls in self.class_priors:
                log_prob = self.class_priors[cls]
                total_words_in_class = self.class_word_totals[cls]
                
                for token in tokens:
                    word_freq = self.word_counts[cls].get(token, 0)
                    log_prob += np.log((word_freq + self.smoothening) / 
                                       (total_words_in_class + self.smoothening * V))
                    
                class_scores[cls] = log_prob
            
            # Assign class with maximum probability
            predictions.append(max(class_scores, key=class_scores.get))
        
        df[predicted_col] = predictions
