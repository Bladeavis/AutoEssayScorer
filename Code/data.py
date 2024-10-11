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

    # Tokens: Total words in the text, including function words: Pronouns, prepositions, conjunctions, articles, auxiliary verbs, particles, etc.
    tokens = [token.text for token in doc if token.is_alpha]

    # Unique words in the text.
    types = set(tokens) # removes duplicate tokens from the list tokens

    # Lexical diversity - Type-Token Ratio (TTR)
    # TTR is calculated as the number of unique words (types) divided by the total number of words (tokens).
    # A higher type_token_ratio indicates a greater variety of words used in the text, which can be an indicator of more complex or varied language
    type_token_ratio = (len(types) / len(tokens)) * 100

    # Lexical density is a measure of the proportion of content words (nouns, verbs, adjectives, and adverbs) to the total number of words (tokens).
    # It is used to assess the complexity and informativeness of the text.
    lexical_tokens = [token for token in doc if
                      token.is_alpha and token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]  # Main lexical words
    lexical_density = (len(lexical_tokens) / len(tokens)) * 100

    return type_token_ratio, lexical_density


# Function to calculate readability scores
def calculate_readability_scores(text):
    textstat.set_lang("de")

    # Flesch Reading Ease The Flesch Reading Ease formula will output a number from 0 to 100 - a higher score
    # indicates easier reading.
    # Score Difficulty: 90-100 Very Easy - 80-89 Easy - 70-79 Fairly Easy - 60-69 Standard
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

# Save the updated dataframe
df.to_csv('/Users/nurhayataltunok/PycharmProjects/CERF-German-Deep Learning/All_Data_with_features.csv', index=False)

# ____________________________________________________________________________________________________________________________________

# Load your dataset
df = pd.read_csv('All_Data_with_features.csv')

# Calculate the frequency of each CEFR level
cefr_frequency = df['Cefr'].value_counts()

# Print the frequency
print(cefr_frequency)

# ____________________________________________________________________________________________________________________________________
# Load your dataset
df = pd.read_csv('All_Data_with_features.csv')

# Filter out 44 elements from the rows where the CEFR level is 'B2'
b2_rows = df[df['Cefr'] == 'B2']
remaining_b2_rows = b2_rows.iloc[44:]  # Keep all but the first 44 rows

# Combine the remaining B2 rows with the rest of the dataset
df_filtered = pd.concat([df[df['Cefr'] != 'B2'], remaining_b2_rows])

# Save the modified dataset
df_filtered.to_csv('All_Data_with_features_new.csv', index=False)

# ____________________________________________________________________________________________________________________________________

# Read the CSV file into a DataFrame
df = pd.read_csv('All_Data_with_features_new.csv')

# Create a new column 'NCefr' by extracting the letter part of the 'Cefr' column
df['NCefr'] = df['Cefr'].str.extract(r'([A-Z]+)')

# Insert the new column before the 'Cefr' column
cols = df.columns.tolist()
cols.insert(cols.index('Cefr'), cols.pop(cols.index('NCefr')))
df = df[cols]

# Save the modified DataFrame back to the CSV file
df.to_csv('All_Data_with_features_new.csv', index=False)

# ____________________________________________________________________________________________________________________________________

# Define the function to count levels in a CSV file
def count_levels(file_path):
    df = pd.read_csv(file_path)
    level_counts = df['Cefr'].value_counts()
    return level_counts


# List of files to process
files = [
    'Whole_Data/DISKO_Testdaf.csv',
    'Whole_Data/Falko_C2.csv',
    'Whole_Data/Elias_A1.csv',
    'Whole_Data/Falko_Mixed.csv',
    'Whole_Data/Merlin.csv'
]

# Dictionary to store the results
results = {}

# Process each file
for file in files:
    level_counts = count_levels(file)
    results[file] = level_counts

# Print the results
for file, counts in results.items():
    print(f"{file}:")
    for level in ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']:
        print(f"  {level} - {counts.get(level, 0)} texts")


# ____________________________________________________________________________________________________________________________________

# Define the function to count levels in a CSV file
def count_levels(file_path):
    df = pd.read_csv(file_path)
    level_counts = df['Cefr'].value_counts()
    return level_counts

# List of files to process
files = [
    'All_Data_with_features.csv',
    'All_Data_with_features_new.csv'
]

# Dictionary to store the results
results = {}

# Process each file
for file in files:
    level_counts = count_levels(file)
    results[file] = level_counts

# Print the results
for file, counts in results.items():
    print(f"{file}:")
    for level in ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']:
        print(f"  {level} - {counts.get(level, 0)} texts")
