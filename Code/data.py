import pandas as pd
import language_tool_python
import spacy
import textstat

# Load SpaCy German model
nlp = spacy.load("de_core_news_sm")

# Load your dataset
df = pd.read_csv('All_Data.csv')

# Initialize lists to store extracted features
grammar_errors_list = []  # Using LanguageTool
type_token_ratio_list = []
lexical_density_list = []
flesch_reading_ease_list = []
wiener_sachtextformel_1_list = []
wiener_sachtextformel_2_list = []
wiener_sachtextformel_3_list = []
wiener_sachtextformel_4_list = []
wiener_sachtextformel_average_list = []

# Function to check grammar using LanguageTool
tool = language_tool_python.LanguageTool('de')


def check_grammar_languagetool(text):
    matches = tool.check(text)
    return len(matches)


# Function to calculate lexical features
def calculate_lexical_features(text):
    doc = nlp(text)

    # Tokens: Total words in the text, including function words: Pronouns, prepositions, conjunctions, articles,
    # auxiliary verbs, particles, etc.
    tokens = [token.text for token in doc if token.is_alpha]

    # Unique words in the text.
    types = set(tokens)

    # Lexical Metrics

    # TTR is calculated as the number of unique words divided by the total number of words.
    type_token_ratio = (len(types) / len(tokens)) * 100

    # Lexical density
    lexical_tokens = [token for token in doc if
                      token.is_alpha and token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]  # Main lexical words
    lexical_density = (len(lexical_tokens) / len(tokens)) * 100

    return type_token_ratio, lexical_density


# Function to calculate readability scores
def calculate_readability_scores(text):
    textstat.set_lang("de")

    # Flesch Reading Ease The Flesch Reading Ease formula will output a number from 0 to 100 - a higher score
    # indicates easier reading. Score Difficulty: 90-100 Very Easy - 80-89 Easy - 70-79 Fairly Easy - 60-69 Standard
    # 50-59 Fairly Difficult - 30-49 Difficult - 0-29 Very Confusing
    flesch_reading_ease = textstat.flesch_reading_ease(text)

    # Wiener Sachtextformel
    # Die Skala beginnt bei Schulstufe 4 und endet bei 15,
    # wobei ab der Stufe 12 eher von Schwierigkeitsstufen als von Schulstufen gesprochen werden sollte.
    # Ein Wert von 4 steht demnach für sehr leichten Text, dagegen bezeichnet 15 einen sehr schwierigen Text.
    wiener_sachtextformel_1 = textstat.wiener_sachtextformel(text, 1)
    wiener_sachtextformel_2 = textstat.wiener_sachtextformel(text, 2)
    wiener_sachtextformel_3 = textstat.wiener_sachtextformel(text, 3)
    wiener_sachtextformel_4 = textstat.wiener_sachtextformel(text, 4)

    return flesch_reading_ease, wiener_sachtextformel_1, wiener_sachtextformel_2, wiener_sachtextformel_3, wiener_sachtextformel_4


# Iterate through the dataset and extract features
for text in df['Text']:
    # Grammar Errors
    grammar_errors = check_grammar_languagetool(text)  # Using LanguageTool
    grammar_errors_list.append(grammar_errors)  # Using LanguageTool

    # Lexical Features
    type_token_ratio, lexical_density = calculate_lexical_features(text)
    type_token_ratio_list.append(type_token_ratio)
    lexical_density_list.append(lexical_density)

    # Readability Scores
    readability_scores = calculate_readability_scores(text)
    flesch_reading_ease_list.append(readability_scores[0])
    wiener_sachtextformel_1_list.append(readability_scores[1])
    wiener_sachtextformel_2_list.append(readability_scores[2])
    wiener_sachtextformel_3_list.append(readability_scores[3])
    wiener_sachtextformel_4_list.append(readability_scores[4])
    wiener_sachtextformel_average_list.append(
        (readability_scores[1] + readability_scores[2] + readability_scores[3] + readability_scores[4]) / 4)

# Add extracted features to the dataframe
df['Grammar_Errors'] = grammar_errors_list  # Using LanguageTool
df['Type_Token_Ratio'] = type_token_ratio_list
df['Lexical_Density'] = lexical_density_list
df['Flesch_Reading_Ease'] = flesch_reading_ease_list
df['Wiener_Sachtextformel_Average'] = wiener_sachtextformel_average_list
# df['Wiener_Sachtextformel_1'] = wiener_sachtextformel_1_list
# df['Wiener_Sachtextformel_2'] = wiener_sachtextformel_2_list
# df['Wiener_Sachtextformel_3'] = wiener_sachtextformel_3_list
# df['Wiener_Sachtextformel_4'] = wiener_sachtextformel_4_list
