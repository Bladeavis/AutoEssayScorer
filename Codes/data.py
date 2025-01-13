#import ast
import pandas as pd
from collections import Counter
import language_tool_python
import spacy
import textstat



# SpaCy German model
nlp = spacy.load("de_core_news_sm")

# Dataset
df = pd.read_csv('DataCombined.csv')

# Initializing lists to store extracted features
grammar_errors_list = []  # Using LanguageTool
whitespace_errors_list = []
typographical_errors_list = []
style_errors_list = []
uncategorized_errors_list = []
misspelling_errors_list = []
type_token_ratio_list = []
lexical_density_list = []
flesch_reading_ease_list = []
wiener_sachtextformel_1_list = []
wiener_sachtextformel_2_list = []
wiener_sachtextformel_3_list = []
wiener_sachtextformel_4_list = []
wiener_sachtextformel_average_list = []
verb_conjugations_list = []
avarage_sentence_length_list = []
unique_tenses_list = []
dep_counts_list = []
tree_depth_list = []
avg_dep_distance_list = []

# -------------------

# LanguageTool for German grammar checking
tool = language_tool_python.LanguageTool('de')


def categorize_errors(text):
    """
    Analyzes text for various types of language errors by categorizing them
    based on predefined error categories. The function processes the given
    input text and applies a language-check tool to detect and count
    specific error types, including 'whitespace', 'style', 'typographical',
    'misspelling', and 'uncategorized'. The total number of errors and
    categorized error counts are returned.

    :param text: Text input to be analyzed for language-related issues.
    :type text: str
    :return: A tuple containing the total number of detected errors
             and a dictionary that maps error categories to their
             respective counts.
    :rtype: tuple[int, dict[str, int]]
    """
    matches = tool.check(text)
    error_categories = {
        'whitespace': 0,
        'style': 0,
        'uncategorized': 0,
        'typographical': 0,
        'misspelling': 0,
      # 'duplication': 0  # It's not much, so extracted from the dataframe
    }
    for match in matches:
        if match.ruleIssueType in error_categories:
            error_categories[match.ruleIssueType] += 1
    total_errors = len(matches)
    return total_errors, error_categories

# -------------------

# Function to calculate lexical features
def calculate_lexical_features(text):
    doc = nlp(text)

    # Tokens: Total words in the text, including function words:
    # Pronouns, prepositions, conjunctions, articles, auxiliary verbs, particles, etc.
    tokens = [token.text for token in doc if token.is_alpha]

    # Unique words in the text. (removing duplicates)
    types = set(tokens)

    # Lexical diversity - Type-Token Ratio (TTR)
    # TTR is calculated as the number of unique words (types) divided by the total number of words (tokens).
    # A higher type_token_ratio indicates a greater variety of words used in the text,
    # which can be an indicator of more complex or varied language
    type_token_ratio = (len(types) / len(tokens)) * 100

    # Lexical density is a measure of the proportion of content words (nouns, verbs, adjectives, and adverbs) to the total number of words (tokens).
    # It is used to assess the complexity and informativeness of the text.
    lexical_tokens = [token for token in doc if
                      token.is_alpha and token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]  # Main lexical words
    lexical_density = (len(lexical_tokens) / len(tokens)) * 100

    return type_token_ratio, lexical_density

# -------------------

# Function to calculate readability scores
def calculate_readability_scores(text):
    textstat.set_lang("de") # Setting language for textstat to German

    # The Flesch Reading Ease formula will output a number from 0 to 100
    # Higher Score = Easier to read
    # Score Difficulty:
    #                   90-100 Very Easy
    #                   80-89 Easy
    #                   70-79 Fairly Easy
    #                   60-69 Standard
    #                   50-59 Fairly Difficult
    #                   30-49 Difficult
    #                   0-29 Very Confusing
    flesch_reading_ease = textstat.flesch_reading_ease(text)

    # Wiener Sachtextformel: scores range from 4 (easy) to 15 (difficult)
    wiener_sachtextformel_1 = textstat.wiener_sachtextformel(text, 1)
    wiener_sachtextformel_2 = textstat.wiener_sachtextformel(text, 2)
    wiener_sachtextformel_3 = textstat.wiener_sachtextformel(text, 3)
    wiener_sachtextformel_4 = textstat.wiener_sachtextformel(text, 4)

    return flesch_reading_ease, wiener_sachtextformel_1, wiener_sachtextformel_2, wiener_sachtextformel_3, wiener_sachtextformel_4

# -------------------

# Function to calculate verb conjugations - Using SpaCy
def calculate_verb_conjugations(text):
    doc = nlp(text)
    verb_tenses = {'present': 0, 'past': 0, 'future': 0, 'conditional': 0, 'subjunctive': 0}
    for i, token in enumerate(doc):
        tense = None
        mood = None
        if token.pos_ == 'VERB' or token.pos_ == 'AUX':
            if token.morph.get('Tense'):
                tense = token.morph.get('Tense')[0]
                if tense == 'Pres':
                    verb_tenses['present'] += 1
                elif tense == 'Past':
                    verb_tenses['past'] += 1
                elif tense == 'Fut':
                    verb_tenses['future'] += 1
            if token.morph.get('Mood'):
                mood = token.morph.get('Mood')[0]
                if mood == 'Cnd':
                    verb_tenses['conditional'] += 1
                elif mood == 'Sub':
                    verb_tenses['subjunctive'] += 1
            # Check for future tense with auxiliary verb "werden"
            if token.lemma_ == 'werden' and i + 1 < len(doc) and doc[i + 1].pos_ == 'VERB':
                verb_tenses['future'] += 1
            #print(f"Token: {token.text}, Tense: {tense}, Mood: {mood}")
    return verb_tenses


# Function to calculate total tenses
def calculate_different_tenses(verb_conjugations):
    different_tenses = sum(1 for tense in verb_conjugations.values() if tense > 0)
    return different_tenses

# Function to calculate average sentence length
def calculate_average_sentence_length(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    total_words = sum(len(sentence) for sentence in sentences)
    average_sentence_length = total_words / len(sentences) if sentences else 0
    return average_sentence_length

# Function to perform dependency analysis
def analyze_text(text):
    doc = nlp(text)

    # Count Dependency Types
    dep_counts = Counter([token.dep_ for token in doc])

    # Depth of Dependency Tree: The maximum depth of the dependency tree in the text.
    def get_tree_depth(token):
        if not list(token.children):
            return 1
        else:
            return 1 + max(get_tree_depth(child) for child in token.children)

    tree_depth = max(get_tree_depth(sent.root) for sent in doc.sents)

    # Average Dependency Distance: Average distance between a token and its head in the dependency tree.
    dep_distances = [abs(token.i - token.head.i) for token in doc if token.dep_ != 'ROOT']
    avg_dep_distance = sum(dep_distances) / len(dep_distances) if dep_distances else 0

    return dep_counts, tree_depth, avg_dep_distance

# -------------------

#Processing each text in the dataset to extract features
for text in df['Text']:
    # Grammar Errors (LanguageTool)
    total_errors, error_categories = categorize_errors(text)
    grammar_errors_list.append(total_errors)
    whitespace_errors_list.append(error_categories['whitespace'])
    typographical_errors_list.append(error_categories['typographical'])
    style_errors_list.append(error_categories['style'])
    uncategorized_errors_list.append(error_categories['uncategorized'])
    misspelling_errors_list.append(error_categories['misspelling'])

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
    # Average Wiener Sachtextformel score
    wiener_sachtextformel_average_list.append(
        (readability_scores[1] + readability_scores[2] + readability_scores[3] + readability_scores[4]) / 4)

    # Verb conjugations
    verb_conjugations = calculate_verb_conjugations(text)
    verb_conjugations_list.append(verb_conjugations)

    # Unique tenses
    different_tenses = calculate_different_tenses(verb_conjugations)
    unique_tenses_list.append(different_tenses)

    # Average sentence length
    average_sentence_length = calculate_average_sentence_length(text)
    avarage_sentence_length_list.append(average_sentence_length)

    # Dependency Analysis
    dep_counts, tree_depth, avg_dep_distance = analyze_text(text)
    dep_counts_list.append(dep_counts)
    tree_depth_list.append(tree_depth)
    avg_dep_distance_list.append(avg_dep_distance)


'''
# ____________________________________________________________________________________________________________________________________
# Adding extracted features to the dataframe
df['Total_Errors'] = grammar_errors_list
df['Whitespace_Errors'] = whitespace_errors_list
df['Typographical_Errors'] = typographical_errors_list
df['Style_Errors'] = style_errors_list
df['Uncategorized_Errors'] = uncategorized_errors_list
df['Misspelling_Errors'] = misspelling_errors_list
df['Type_Token_Ratio'] = type_token_ratio_list
df['Lexical_Density'] = lexical_density_list
df['Flesch_Reading_Ease'] = flesch_reading_ease_list
df['Wiener_Sachtextformel_Average'] = wiener_sachtextformel_average_list
df['Average_Sentence_Length'] = avarage_sentence_length_list
df['Verb_Conjugations'] = verb_conjugations_list
df['Unique_Tenses'] = unique_tenses_list
df['Dependency_Counts'] = dep_counts_list
df['Max_Tree_Depth'] = tree_depth_list
df['Avg_Dependency_Distance'] = avg_dep_distance_list

all_dep_types = set(dep for dep_counts in dep_counts_list for dep in dep_counts)
for dep_type in all_dep_types:
    df[f'Dep_{dep_type}'] = [dep_counts.get(dep_type, 0) for dep_counts in dep_counts_list]


# Save the detailed DataFrame to a new CSV file
df.to_csv('DataFeaturesDet.csv', index=False)

# ____________________________________________________________________________________________________________________________________
# Calculate the new Total_Errors
df['Total_Errors'] = df['Total_Errors'] - (df['Whitespace_Errors'] + df['Typographical_Errors'])

# Dropping the specified columns
df.drop(columns=['Whitespace_Errors', 'Typographical_Errors', 'Style_Errors', 'Uncategorized_Errors', 'Misspelling_Errors'], inplace=True)

# Saving the cleaned DataFrame to a new CSV file
df.to_csv('DataClean.csv', index=False)

# ____________________________________________________________________________________________________________________________________


# Function to print error types
def print_error_types(text):
    matches = tool.check(text)
    error_types = set()
    for match in matches:
        error_types.add(match.ruleIssueType)
    return error_types


# Unique error types found:
# whitespace
# style
# uncategorized
# typographical
# misspelling
# duplication

#-------------------

# Whitespace Errors and Typographical Errors are not considered in the error types. 
# They are extracted from the Total_Errors column, these errors cannot be considered as right, 
# because the text in the dataset created with those unnecessary whitespaces. They are not real errors from the language learners.

# Examples of the errors:

Error: Vor dem Punkt sollte kein Leerzeichen stehen.
Context: ...en 44 % Haushalts drei und mehr Personen . In 2015 war es wenig - nur 24 % . Wie s...
Rule: COMMA_PARENTHESIS_WHITESPACE
Category: whitespace
----------------------------------------
Error: Vor dem Punkt sollte kein Leerzeichen stehen.
Context: ...rsonen . In 2015 war es wenig - nur 24 % . Wie sieht ein Deutsch Haushalt in 2015 ...
Rule: COMMA_PARENTHESIS_WHITESPACE
Category: whitespace
----------------------------------------
Error: Eines der beiden Leerzeichen scheint überflüssig.
Context: ... Diese " change " hat eine Frage gemacht , " Sollte man alleine oder mit anderen zusammen l...
Rule: LEERZEICHEN_NACH_VOR_ANFUEHRUNGSZEICHEN
Category: typographical
----------------------------------------
Error: Möglicherweise passen das Nomen und die Wörter, die das Nomen beschreiben, grammatisch nicht zusammen.
Context: ...shalt in 2015 ? 42 % Haushalts sind nur ein Person . Die Zeit es nicht die Gleich . Diese ...
Rule: DE_AGREEMENT
Category: uncategorized
----------------------------------------
Error: Möglicher Tippfehler gefunden.
Context: ...on 1975 bis 1995 , die Haushalts sind " even " bekommen . Ein-dritte mit einen Perso...
Rule: GERMAN_SPELLER_RULE
Category: misspelling
----------------------------------------
Error: Meinten Sie „einem“? (Alternativ prüfen Sie, ob ‚mit‘ durch ‚mir‘ ersetzt werden muss.)
Context: ...sind " even " bekommen . Ein-dritte mit einen Person , ein-dritte mit zwei und ein-dr...
Rule: DEN_DEM
Category: uncategorized

'''
